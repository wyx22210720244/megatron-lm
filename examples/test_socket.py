# import socket
#
# # 获取主机名
# hostname = socket.gethostname()
# print(f"Hostname: {hostname}")
#
# # 解析主机名为IP地址
# ip_address = socket.gethostbyname(hostname)
# print(f"IP Address: {ip_address}")
#
# # 解析特定的主机名为IP地址
# pod_name = "wyx-master-0"
# try:
#     pod_ip = socket.gethostbyname(pod_name)
#     print(f"IP Address for {pod_name}: {pod_ip}")
# except socket.error as err:
#     print(f"Error resolving {pod_name}: {err}")
import shutil
from concurrent.futures import ProcessPoolExecutor

import torch
import torch.distributed as dist
import os
import time
import pathlib
import logging
from typing import List, Optional, Union

from tqdm import tqdm

logger = logging.getLogger(__name__)


class RDMADistConfig:
    """
    Attributes:
        source_dir:         directory to copy from in node 0.
        destination_dir:    directory to copy to in other nodes.
        transit_dir:        if not None, copy from source_dir to transit_dir in node 0 before distribution
                            for better read performance.
        files:              default is '*', copy all files in source_dir.
                            or pass a list of file paths, which is concatenated with source_dir to determine
                            the actual files to copy.
        infer_dest_files:   if True, `files` argument in other nodes are inferred from the value in node 0.
        fullpaths:          if set, overwrites any above logic and this is used as final paths in all nodes.
        concurrent_send:    how many receivers to send files to concurrently for each sender.
                            a good default is provided.
        tcp_store_port:     port listened on MASTER_ADDR for node communication.
        rdma_port:          port for actual file sending and receiving. on the sender side,
                            use ports in range [rdma_port, (rdma_port + NUM_RDMA_NIC // concurrent_send)].
    Constants:
        P2P_BANDWIDTH_GBPS: empirical bandwidth for a single p2p transport.
        RDMA_NIC_BANDWIDTH: rdma capacity with an estimated discount.
        NUM_RDMA_NIC:       number of rdma nics on each node (hardcoded as eth1, eth2, ...).
    """

    def __init__(self,
                 source_dir: str,
                 destination_dir: str,
                 files: List[str] = ["*"],
                 transit_dir: Optional[str] = None,
                 infer_dest_files: bool = True,
                 fullpaths: Optional[List[str]] = None,
                 concurrent_send: Optional[int] = None,
                 tcp_store_port: int = 31000,
                 rdma_port: int = 31000):
        self.source_dir = source_dir
        self.destination_dir = destination_dir
        self.files = files
        self.transit_dir = transit_dir
        self.infer_dest_files = infer_dest_files
        self.fullpaths = fullpaths
        self.concurrent_send = concurrent_send
        self.tcp_store_port = tcp_store_port
        self.rdma_port = rdma_port

        self.P2P_BANDWIDTH_GBPS = 2.5
        self.RDMA_NIC_BANDWIDTH = 200 / 8 * 0.9
        self.NUM_RDMA_NIC = 4


def get_nic_ip(nic="eth1"):
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(('8.8.8.8', 1))
    return s.getsockname()[0]


def receive_file(config: RDMADistConfig, filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    file_size_tensor = torch.zeros(1, dtype=torch.long)
    dist.recv(file_size_tensor, src=0)
    file_size = file_size_tensor.item()

    buffer = torch.empty(file_size, dtype=torch.uint8)
    dist.recv(buffer, src=0)

    with open(filepath, 'wb') as f:
        f.write(buffer.numpy().tobytes())

    logger.info(f"Received file {filepath}")


def receive_files(config: RDMADistConfig):
    for f in config.fullpaths:
        receive_file(config, os.path.join(config.destination_dir, f))


def send_file(config: RDMADistConfig, filepath: str):
    file_size = os.path.getsize(filepath)
    file_size_tensor = torch.tensor([file_size], dtype=torch.long)

    with open(filepath, 'rb') as f:
        data = f.read()
        buffer = torch.tensor(bytearray(data), dtype=torch.uint8)

    for rank in range(1, dist.get_world_size()):
        dist.send(file_size_tensor, dst=rank)
        dist.send(buffer, dst=rank)

    logger.info(f"Sent file {filepath} to all nodes")


def send_files(config: RDMADistConfig):
    for f in config.fullpaths:
        send_file(config, os.path.join(config.source_dir, f))


def distribute_file_rdma(config: Union[RDMADistConfig, dict]):
    if isinstance(config, dict):
        config = RDMADistConfig(**config)

    assert dist.is_initialized(), "distribution of file requires pytorch distributed to be initialized"

    node_rank = dist.get_rank() // torch.cuda.device_count()

    if dist.get_rank() % torch.cuda.device_count() != 0:
        dist.barrier()
        return

    if node_rank == 0 and config.fullpaths is None:
        source = pathlib.Path(config.source_dir)
        config.files = list(
            set([os.path.relpath(p, source) for f in config.files for p in source.rglob(f) if p.is_file()])
        )
        logger.info(f"found all files in {config.source_dir}: {len(config.files)} in total.")
        config.fullpaths = [os.path.abspath(os.path.join(config.source_dir, f)) for f in config.files]
        if config.transit_dir:
            logger.info(f"[{node_rank}] copying files from {config.source_dir} to transit dir {config.transit_dir}")
            with ProcessPoolExecutor(max_workers=10) as executor:
                list(
                    tqdm(
                        executor.map(
                            lambda x: shutil.copy2(*x),
                            [(os.path.join(config.source_dir, file), os.path.join(config.transit_dir, file)) for file in
                             config.files],
                        ),
                        total=len(config.files),
                    )
                )
            logger.info(f"[{node_rank}] copy finished")
            config.fullpaths = [os.path.abspath(os.path.join(config.transit_dir, f)) for f in config.files]

    if node_rank == 0:
        total_size_gb = sum([os.path.getsize(p) for p in config.fullpaths]) / 1024 / 1024 / 1024
        logger.info(f"total size to transport: {total_size_gb} GB")

    current_ip = get_nic_ip("eth1")
    logger.info(f"[{node_rank}] current ip: {current_ip}")

    if node_rank == 0:
        node_0_files = config.files
    else:
        node_0_files = [None] * len(config.files)

    dist.broadcast_object_list(node_0_files, src=0)

    if node_rank != 0:
        config.files = node_0_files
        config.fullpaths = [os.path.abspath(os.path.join(config.destination_dir, f)) for f in config.files]

    if node_rank == 0:
        send_files(config)
    else:
        receive_files(config)

    dist.barrier()
    logger.info("file distributed across all nodes")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.DEBUG,
    )
    dist.init_process_group(backend="nccl")
    start = time.time()
    megatron_root = "/root/Megatron-LM/checkpoints/gpt2"
    # n_iter = open(os.path.join(megatron_root, "latest_checkpointed_iteration.txt")).read()
    # folder = os.path.join(megatron_root, f"iter_{n_iter.zfill(7)}")
    distribute_file_rdma(
        RDMADistConfig(
            source_dir=megatron_root,
            destination_dir=megatron_root,
            concurrent_send=2,
            files=["rdma_test_file.txt"],
        )
    )
    if dist.get_rank() == 0:
        logger.info(f"file distribution time {time.time() - start}s")
    dist.destroy_process_group()

