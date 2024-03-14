# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Model and data parallel groups."""

import os
from typing import Optional
import json
import torch
from megatron import get_args
from .utils import GlobalMemoryBuffer
from collections import defaultdict, Counter

# Intra-layer model parallel group that the current rank belongs to.
_TENSOR_MODEL_PARALLEL_GROUP = None
# Inter-layer model parallel group that the current rank belongs to.
_PIPELINE_MODEL_PARALLEL_GROUP = {}
_PIPELINE_MODEL_PARALLEL_GROUP_DICT = None
# Model parallel group (both intra- and pipeline) that the current rank belongs to.
_MODEL_PARALLEL_GROUP = None
# Embedding group.
_EMBEDDING_GROUP = {}
# Position embedding group.
_POSITION_EMBEDDING_GROUP = {}
# Data parallel group that the current rank belongs to.
_DATA_PARALLEL_GROUP = {}
_DATA_PARALLEL_IDX = 0
_DATA_PARALLEL_GROUP_GLOO = None
# tensor model parallel group and data parallel group combined
# used for fp8 and moe training
_TENSOR_AND_DATA_PARALLEL_GROUP = None
# Expert parallel group that the current rank belongs to.
_TENSOR_AND_EXPERT_PARALLEL_GROUP = None
_DATA_MODULO_EXPERT_PARALLEL_GROUP = None

_VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = None
_VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
_PIPELINE_MODEL_PARALLEL_SPLIT_RANK = None

# These values enable us to change the mpu sizes on the fly.
_MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = None
_MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
_MPU_TENSOR_MODEL_PARALLEL_RANK = None
_MPU_PIPELINE_MODEL_PARALLEL_RANK = None

# A list of ranks that have a copy of the embedding.
_EMBEDDING_GLOBAL_RANKS = {}
_EMBEDDING_IDX = 0
_EMBEDDING_RANKS_IDX = 0
# A list of ranks that have a copy of the position embedding.
_POSITION_EMBEDDING_GLOBAL_RANKS = {}
_POSITION_EMBEDDING_IDX = 0
_POSITION_EMBEDDING_RANKS_IDX = 0
# A list of global ranks for each pipeline group to ease calculation of the source
# rank when broadcasting from the first or last pipeline stage.
_PIPELINE_IDX = 0
_PIPELINE_GLOBAL_RANKS = {}
_PIPELINE_GLOBAL_RANKS_DICT = None

_TENSOR_PARALLEL_RANKS = None
# A list of global ranks for each data parallel group to ease calculation of the source
# rank when broadcasting weights from src to all other data parallel ranks
_DATA_PARALLEL_GLOBAL_RANKS = {}
_DATA_PARALLEL_GLOBAL_LAYERS = {}
_DATA_PARALLEL_NUM_LAYER = []
_DATA_PARALLEL_OFFSETS = []
# _DATA_PARALLEL_GLOBAL_RANKS_IDX = 0

# Context parallel group that the current rank belongs to
_CONTEXT_PARALLEL_GROUP = None
# A list of global ranks for each context parallel group to ease calculation of the
# destination rank when exchanging KV/dKV between context parallel_ranks
_CONTEXT_PARALLEL_GLOBAL_RANKS = None

# Data parallel group information with context parallel combined.
_DATA_PARALLEL_GROUP_WITH_CP = None
_DATA_PARALLEL_GROUP_WITH_CP_GLOO = None
_DATA_PARALLEL_GLOBAL_RANKS_WITH_CP = None

# combined parallel group of TP, DP, and CP used for fp8
_TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP = None

# Memory buffers to avoid dynamic memory allocation
_GLOBAL_MEMORY_BUFFER = None


def get_nccl_options(pg_name, nccl_comm_cfgs):
    """Set the NCCL process group options.

    Arguments:
        pg_name (str): process group name
        nccl_comm_cfgs (dict): nccl communicator configurations

    When an option (e.g., max_ctas) is not found in the config, use the NCCL default setting.
    """
    if pg_name in nccl_comm_cfgs:
        nccl_options = torch.distributed.ProcessGroupNCCL.Options()
        nccl_options.config.cga_cluster_size = nccl_comm_cfgs[pg_name].get('cga_cluster_size', 4)
        nccl_options.config.max_ctas = nccl_comm_cfgs[pg_name].get('max_ctas', 32)
        nccl_options.config.min_ctas = nccl_comm_cfgs[pg_name].get('min_ctas', 1)
        return nccl_options
    else:
        return None


def initialize_model_parallel(
        tensor_model_parallel_size: int = 1,
        pipeline_model_parallel_size: int = 1,
        virtual_pipeline_model_parallel_size: Optional[int] = None,
        pipeline_model_parallel_split_rank: Optional[int] = None,
        use_sharp: bool = False,
        context_parallel_size: int = 1,
        expert_model_parallel_size: int = 1,
        nccl_communicator_config_path: Optional[str] = None,
) -> None:
    """Initialize model data parallel groups.

    """
    args = get_args()
    layer_allocation = json.load(open("layers.json"))
    group_allocation = json.load(open("allocations.json"))
    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized()
    world_size: int = torch.distributed.get_world_size()
    gpu_count: int = 0
    data_parallel_size: int = 0
    for key, value in group_allocation.items():
        for gpu in value:
            if isinstance(gpu, list):
                gpu_count += len(gpu)
            else:
                gpu_count += 1
        data_parallel_size += 1
    args.data_parallel_size = data_parallel_size
    # world_size的检查
    if world_size != gpu_count:
        raise RuntimeError(
            f"world_size ({world_size}) is not equal to gpu_count ({gpu_count}) in gpu_allocation"
        )

    if data_parallel_size % expert_model_parallel_size != 0:
        raise RuntimeError(
            f"data_parallel_size ({data_parallel_size}) is not divisible by expert_model_parallel_size "
        )

    if expert_model_parallel_size > 1 and context_parallel_size > 1:
        raise RuntimeError(
            f"combination of expert model prallellism and context parallelism is not supported"
        )

    # groups数现由算法直接生成
    # num_tensor_model_parallel_groups: int = world_size // tensor_model_parallel_size
    # num_pipeline_model_parallel_groups: int = world_size // pipeline_model_parallel_size

    if virtual_pipeline_model_parallel_size is not None:
        if not pipeline_model_parallel_size > 2:
            raise RuntimeError(
                "pipeline-model-parallel size should be greater than 2 with interleaved schedule"
            )
        global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
        global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
        _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = 0
        _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = virtual_pipeline_model_parallel_size

    if pipeline_model_parallel_split_rank is not None:
        global _PIPELINE_MODEL_PARALLEL_SPLIT_RANK
        _PIPELINE_MODEL_PARALLEL_SPLIT_RANK = pipeline_model_parallel_split_rank

    # 当前rank的全局号
    rank = torch.distributed.get_rank()

    nccl_comm_cfgs = {}
    if nccl_communicator_config_path is not None:
        try:
            import yaml
        except ImportError:
            raise RuntimeError(
                "Cannot import `yaml`. Setting custom nccl communicator configs "
                "requires the yaml package."
            )

        with open(nccl_communicator_config_path, "r") as stream:
            nccl_comm_cfgs = yaml.safe_load(stream)

    # Build the data-parallel groups.
    global _DATA_PARALLEL_GROUP
    global _DATA_PARALLEL_IDX
    global _DATA_PARALLEL_GROUP_GLOO
    global _DATA_PARALLEL_GLOBAL_RANKS
    global _DATA_PARALLEL_GLOBAL_LAYERS
    global _DATA_PARALLEL_GROUP_WITH_CP
    global _DATA_PARALLEL_GROUP_WITH_CP_GLOO
    global _DATA_PARALLEL_GLOBAL_RANKS_WITH_CP
    global _DATA_PARALLEL_NUM_LAYER
    global _DATA_PARALLEL_OFFSETS
    assert not _DATA_PARALLEL_GROUP, 'data parallel group is already initialized'
    all_data_parallel_group_ranks_with_cp = []
    for key, value in group_allocation.items():
        for gpus in value:
            tp_size = len(gpus)
    unique_layer_allocation = set()
    # unique_layer_allocation.add(0)
    for _, layers in layer_allocation.items():
        for layer in layers:
            unique_layer_allocation.add(layer)
    sorted_unique_intervals = sorted(list(unique_layer_allocation))
    # print(f"sorted_unique_intervals:{sorted_unique_intervals}")
    if tp_size ==2:
        dp_group_comm_up = defaultdict(list)
        dp_group_comm_down = defaultdict(list)
        group_idx = 0
        for layer in sorted_unique_intervals:
            for dp_group, gpus in group_allocation.items():
                for idx, gpu in enumerate(gpus):
                    if layer_allocation[dp_group][idx] >= layer:
                        dp_group_comm_up[group_idx].append(gpu[0])
                        dp_group_comm_down[group_idx].append(gpu[1])
                        break
            group_idx += 1
        dp_group_comm = defaultdict(list)
        for key, _ in dp_group_comm_up.items():
            dp_group_comm[key].append(dp_group_comm_up[key])
            dp_group_comm[key].append(dp_group_comm_down[key])
        print(f"dp_group_comm:{dp_group_comm}")
    if tp_size ==4:
        dp_group_comm_1 = defaultdict(list)
        dp_group_comm_2 = defaultdict(list)
        dp_group_comm_3 = defaultdict(list)
        dp_group_comm_4 = defaultdict(list)
        group_idx = 0
        for layer in sorted_unique_intervals:
            for dp_group, gpus in group_allocation.items():
                for idx, gpu in enumerate(gpus):
                    if layer_allocation[dp_group][idx] >= layer:
                        dp_group_comm_1[group_idx].append(gpu[0])
                        dp_group_comm_2[group_idx].append(gpu[1])
                        dp_group_comm_3[group_idx].append(gpu[2])
                        dp_group_comm_4[group_idx].append(gpu[3])
                        break
            group_idx += 1
        dp_group_comm = defaultdict(list)
        for key, _ in dp_group_comm_1.items():
            dp_group_comm[key].append(dp_group_comm_1[key])
            dp_group_comm[key].append(dp_group_comm_2[key])
            dp_group_comm[key].append(dp_group_comm_3[key])
            dp_group_comm[key].append(dp_group_comm_4[key])
    if tp_size ==8:
        dp_group_comm_1 = defaultdict(list)
        dp_group_comm_2 = defaultdict(list)
        dp_group_comm_3 = defaultdict(list)
        dp_group_comm_4 = defaultdict(list)
        dp_group_comm_5 = defaultdict(list)
        dp_group_comm_6 = defaultdict(list)
        dp_group_comm_7 = defaultdict(list)
        dp_group_comm_8 = defaultdict(list)
        group_idx = 0
        for layer in sorted_unique_intervals:
            for dp_group, gpus in group_allocation.items():
                for idx, gpu in enumerate(gpus):
                    if layer_allocation[dp_group][idx] >= layer:
                        dp_group_comm_1[group_idx].append(gpu[0])
                        dp_group_comm_2[group_idx].append(gpu[1])
                        dp_group_comm_3[group_idx].append(gpu[2])
                        dp_group_comm_4[group_idx].append(gpu[3])
                        dp_group_comm_5[group_idx].append(gpu[4])
                        dp_group_comm_6[group_idx].append(gpu[5])
                        dp_group_comm_7[group_idx].append(gpu[6])
                        dp_group_comm_8[group_idx].append(gpu[7])
                        break
            group_idx += 1
        dp_group_comm = defaultdict(list)
        for key, _ in dp_group_comm_1.items():
            dp_group_comm[key].append(dp_group_comm_1[key])
            dp_group_comm[key].append(dp_group_comm_2[key])
            dp_group_comm[key].append(dp_group_comm_3[key])
            dp_group_comm[key].append(dp_group_comm_4[key])
            dp_group_comm[key].append(dp_group_comm_5[key])
            dp_group_comm[key].append(dp_group_comm_6[key])
            dp_group_comm[key].append(dp_group_comm_7[key])
            dp_group_comm[key].append(dp_group_comm_8[key])
    dp_layer_comm = []
    sorted_unique_intervals.insert(0, 0)
    for idx in range(1, len(sorted_unique_intervals)):
        dp_layer_comm.append(sorted_unique_intervals[idx] - sorted_unique_intervals[idx - 1])
    # 每个DP组中，每个rank的层数
    num_layer = defaultdict(list)
    for dp_group, layers in layer_allocation.items():
        pre = 0
        for idx, layer in enumerate(layers):
            num_layer[dp_group].append(layer-pre)
            pre = layer
    for dp_group, gpus in group_allocation.items():
        for idx, gpu in enumerate(gpus):
            if rank in gpu :
                _DATA_PARALLEL_NUM_LAYER = num_layer[dp_group][idx]
                if idx == 0:
                    _DATA_PARALLEL_OFFSETS = 0
                else:
                    _DATA_PARALLEL_OFFSETS = layer_allocation[dp_group][idx-1]-1
    for dp_group, gpus in dp_group_comm.items():
        for gpu in gpus:
            ranks = gpu
            group = torch.distributed.new_group(
                ranks, pg_options=get_nccl_options('dp', nccl_comm_cfgs)
            )
            group_gloo = torch.distributed.new_group(ranks, backend="gloo")
            if rank in ranks:
                _DATA_PARALLEL_GROUP[_DATA_PARALLEL_IDX] = group
                _DATA_PARALLEL_GROUP_GLOO = group_gloo
                _DATA_PARALLEL_GLOBAL_RANKS[_DATA_PARALLEL_IDX] = ranks
                _DATA_PARALLEL_GLOBAL_LAYERS[_DATA_PARALLEL_IDX] = int(dp_layer_comm[dp_group])
                _DATA_PARALLEL_GROUP_WITH_CP = group
                _DATA_PARALLEL_GROUP_WITH_CP_GLOO = group_gloo
                _DATA_PARALLEL_GLOBAL_RANKS_WITH_CP = ranks
                _DATA_PARALLEL_IDX += 1

    # for i in range(pipeline_model_parallel_size):
    #     start_rank = i * num_pipeline_model_parallel_groups
    #     end_rank = (i + 1) * num_pipeline_model_parallel_groups
    #     for j in range(context_parallel_size * tensor_model_parallel_size):
    #         ranks = range(
    #             start_rank + j, end_rank, context_parallel_size * tensor_model_parallel_size
    #         )
    #         group = torch.distributed.new_group(
    #             ranks, pg_options=get_nccl_options('dp', nccl_comm_cfgs)
    #         )
    #         group_gloo = torch.distributed.new_group(ranks, backend="gloo")
    #         if rank in ranks:
    #             _DATA_PARALLEL_GROUP = group
    #             _DATA_PARALLEL_GROUP_GLOO = group_gloo
    #             _DATA_PARALLEL_GLOBAL_RANKS = ranks
    #     for j in range(tensor_model_parallel_size):
    #         ranks_with_cp = range(start_rank + j, end_rank, tensor_model_parallel_size)
    #         all_data_parallel_group_ranks_with_cp.append(list(ranks_with_cp))
    #         group_with_cp = torch.distributed.new_group(
    #             ranks_with_cp, pg_options=get_nccl_options('dp_cp', nccl_comm_cfgs)
    #         )
    #         group_with_cp_gloo = torch.distributed.new_group(ranks_with_cp, backend="gloo")
    #         if rank in ranks_with_cp:
    #             _DATA_PARALLEL_GROUP_WITH_CP = group_with_cp
    #             _DATA_PARALLEL_GROUP_WITH_CP_GLOO = group_with_cp_gloo
    #             _DATA_PARALLEL_GLOBAL_RANKS_WITH_CP = ranks_with_cp

    # Apply SHARP to DP process groups
    if use_sharp:
        if rank == 0:
            print(
                "The number of process groups to use SHARP with depends on the type "
                "of the network switch. Nvidia QM1 switch supports SAHRP up to 8 "
                "process groups and QM2 supports up to 256 process groups. We apply "
                "SHARP to the communications of the data-parallel domain. If the "
                "number of data-parallel process groups is larger than the max "
                "process groups that the network switch supports, the communication "
                "will fall back to non-SHARP operators. To enable SHARP, "
                "`#SBATCH_NETWORK=sharp` should be set in the sbatch script."
            )
        torch.distributed.barrier(
            group=get_data_parallel_group(with_context_parallel=context_parallel_size > 1),
            device_ids=[torch.cuda.current_device()],
        )
        # Set `NCCL_SHARP_DISABLE=1` to restrict SHARP application to DP process groups
        os.environ["NCCL_SHARP_DISABLE"] = "1"

    # Build the context-parallel groups.
    global _CONTEXT_PARALLEL_GROUP
    global _CONTEXT_PARALLEL_GLOBAL_RANKS
    assert _CONTEXT_PARALLEL_GROUP is None, 'context parallel group is already initialized'
    for key, value in group_allocation.items():
        for gpu in value:
            if isinstance(gpu, list):
                for j in gpu:
                    ranks = [j]
                    group = torch.distributed.new_group(
                        ranks, pg_options=get_nccl_options('cp', nccl_comm_cfgs)
                    )
                    if rank in ranks:
                        _CONTEXT_PARALLEL_GROUP = group
                        _CONTEXT_PARALLEL_GLOBAL_RANKS = ranks
            else:
                ranks = [gpu]
                group = torch.distributed.new_group(
                    ranks, pg_options=get_nccl_options('cp', nccl_comm_cfgs)
                )
                if rank in ranks:
                    _CONTEXT_PARALLEL_GROUP = group
                    _CONTEXT_PARALLEL_GLOBAL_RANKS = ranks
    # for i in range(pipeline_model_parallel_size):
    #     for j in range(data_parallel_size):
    #         start_rank = (
    #                 i * num_pipeline_model_parallel_groups
    #                 + j * tensor_model_parallel_size * context_parallel_size
    #         )
    #         end_rank = (
    #                 i * num_pipeline_model_parallel_groups
    #                 + (j + 1) * tensor_model_parallel_size * context_parallel_size
    #         )
    #         for k in range(tensor_model_parallel_size):
    #             ranks = range(start_rank + k, end_rank, tensor_model_parallel_size)
    #             group = torch.distributed.new_group(
    #                 ranks, pg_options=get_nccl_options('cp', nccl_comm_cfgs)
    #             )
    #             if rank in ranks:
    #                 _CONTEXT_PARALLEL_GROUP = group
    #                 _CONTEXT_PARALLEL_GLOBAL_RANKS = ranks

    # Build the model-parallel groups.
    global _MODEL_PARALLEL_GROUP
    assert _MODEL_PARALLEL_GROUP is None, 'model parallel group is already initialized'
    for _, value in group_allocation.items():
        model_ranks = []
        for gpu in value:
            if isinstance(gpu, list):
                model_ranks.extend(gpu)
            else:
                model_ranks.append(gpu)
        group = torch.distributed.new_group(
            model_ranks, pg_options=get_nccl_options('mp', nccl_comm_cfgs)
        )
        if rank in model_ranks:
            _MODEL_PARALLEL_GROUP = group

    # Build the tensor model-parallel groups.
    global _TENSOR_MODEL_PARALLEL_GROUP
    global _TENSOR_PARALLEL_RANKS
    assert (
            _TENSOR_MODEL_PARALLEL_GROUP is None
    ), 'tensor model parallel group is already initialized'
    global _TENSOR_AND_EXPERT_PARALLEL_GROUP
    assert (
            _TENSOR_AND_EXPERT_PARALLEL_GROUP is None
    ), 'Tensor + expert parallel group is already initialized'
    global _DATA_MODULO_EXPERT_PARALLEL_GROUP
    assert (
            _DATA_MODULO_EXPERT_PARALLEL_GROUP is None
    ), 'Data modulo expert group is already initialized'
    for key, value in group_allocation.items():
        for gpu in value:
            if isinstance(gpu, list):
                tensor_ranks = gpu
                group = torch.distributed.new_group(
                    tensor_ranks, pg_options=get_nccl_options('tp', nccl_comm_cfgs)
                )
            else:
                tensor_ranks = [gpu]
                group = torch.distributed.new_group(
                    tensor_ranks, pg_options=get_nccl_options('tp', nccl_comm_cfgs)
                )
            if rank in tensor_ranks:
                _TENSOR_MODEL_PARALLEL_GROUP = group
                _TENSOR_PARALLEL_RANKS = tensor_ranks
                _TENSOR_AND_EXPERT_PARALLEL_GROUP = group

    # Build the pipeline model-parallel groups and embedding groups
    # (first and last rank in each pipeline model-parallel group).
    global _PIPELINE_MODEL_PARALLEL_GROUP
    global _PIPELINE_IDX
    global _PIPELINE_GLOBAL_RANKS
    assert (
        not _PIPELINE_MODEL_PARALLEL_GROUP
        # _PIPELINE_MODEL_PARALLEL_GROUP is None
    ), 'pipeline model parallel group is already initialized'
    global _EMBEDDING_GROUP
    global _EMBEDDING_IDX
    global _EMBEDDING_GLOBAL_RANKS
    global _EMBEDDING_RANKS_IDX
    assert not _EMBEDDING_GROUP, 'embedding group is already initialized'
    global _POSITION_EMBEDDING_GROUP
    global _POSITION_EMBEDDING_IDX
    global _POSITION_EMBEDDING_GLOBAL_RANKS
    global _POSITION_EMBEDDING_RANKS_IDX
    assert not _POSITION_EMBEDDING_GROUP, 'position embedding group is already initialized'
    for key, value in group_allocation.items():
        pp_list = value
        pp_list = change_to_symmetric_list(pp_list)
        # print(f"当前的rank是{rank},当前的pp_list是{pp_list}\n")
        for idx, pipeline_ranks in enumerate(zip(*pp_list)):
            group = torch.distributed.new_group(
                pipeline_ranks, pg_options=get_nccl_options('pp', nccl_comm_cfgs)
            )
            if rank in pipeline_ranks:
                print(f"当前的rank为{rank},当前的pipeline ranks为{pipeline_ranks}")
                _PIPELINE_MODEL_PARALLEL_GROUP[_PIPELINE_IDX] = group
                _PIPELINE_GLOBAL_RANKS[_PIPELINE_IDX] = pipeline_ranks
                _PIPELINE_IDX += 1

            # Setup embedding group (to exchange gradients between
            # first and last stages).
            if len(pipeline_ranks) > 1:
                embedding_ranks = [pipeline_ranks[0], pipeline_ranks[-1]]
                position_embedding_ranks = [pipeline_ranks[0]]
                if pipeline_model_parallel_split_rank is not None:
                    if pipeline_ranks[pipeline_model_parallel_split_rank] not in embedding_ranks:
                        embedding_ranks = [
                            pipeline_ranks[0],
                            pipeline_ranks[pipeline_model_parallel_split_rank],
                            pipeline_ranks[-1],
                        ]
                    if pipeline_ranks[pipeline_model_parallel_split_rank] not in position_embedding_ranks:
                        position_embedding_ranks = [pipeline_ranks[0],
                                                    pipeline_ranks[pipeline_model_parallel_split_rank]]
            else:
                # 没有PP组的情况
                embedding_ranks = pipeline_ranks
                position_embedding_ranks = pipeline_ranks

            group = torch.distributed.new_group(
                embedding_ranks, pg_options=get_nccl_options('embd', nccl_comm_cfgs)
            )
            if rank in embedding_ranks:
                _EMBEDDING_GROUP[_EMBEDDING_IDX] = group
                _EMBEDDING_IDX += 1
            if rank in pipeline_ranks:
                _EMBEDDING_GLOBAL_RANKS[_EMBEDDING_RANKS_IDX] = embedding_ranks
                _EMBEDDING_RANKS_IDX += 1
            group = torch.distributed.new_group(
                position_embedding_ranks, pg_options=get_nccl_options('embd', nccl_comm_cfgs)
            )
            if rank in position_embedding_ranks:
                _POSITION_EMBEDDING_GROUP[_POSITION_EMBEDDING_IDX] = group
                _POSITION_EMBEDDING_IDX += 1
            if rank in pipeline_ranks:
                _POSITION_EMBEDDING_GLOBAL_RANKS[_POSITION_EMBEDDING_RANKS_IDX] = position_embedding_ranks
                _POSITION_EMBEDDING_RANKS_IDX += 1
    print(f"当前rank为{torch.distributed.get_rank()},当前的pp group为{_PIPELINE_MODEL_PARALLEL_GROUP}")
    # Build the tensor + data parallel groups.
    global _TENSOR_AND_DATA_PARALLEL_GROUP
    global _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP
    assert (
            _TENSOR_AND_DATA_PARALLEL_GROUP is None
    ), 'Tensor + data parallel group is already initialized'
    # tensor_and_data_group_size_with_cp: int = tensor_model_parallel_size * data_parallel_size * context_parallel_size
    # num_tensor_and_data_groups_with_cp: int = world_size // tensor_and_data_group_size_with_cp
    # for i in range(num_tensor_and_data_groups_with_cp):
    #     start_rank = i * tensor_and_data_group_size_with_cp
    #     end_rank = start_rank + tensor_and_data_group_size_with_cp
    #     ranks = range(start_rank, end_rank)
    #     group = torch.distributed.new_group(
    #         ranks, pg_options=get_nccl_options('tp_dp_cp', nccl_comm_cfgs)
    #     )
    #     if rank in ranks:
    #         _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP = group
    #
    #     for j in range(context_parallel_size):
    #         ranks = []
    #         for k in range(data_parallel_size):
    #             start_rank = (
    #                     i * tensor_and_data_group_size_with_cp
    #                     + j * tensor_model_parallel_size
    #                     + k * tensor_model_parallel_size * context_parallel_size
    #             )
    #             end_rank = start_rank + tensor_model_parallel_size
    #             ranks = ranks + list(range(start_rank, end_rank))
    #         group = torch.distributed.new_group(
    #             ranks, pg_options=get_nccl_options('tp_dp', nccl_comm_cfgs)
    #         )
    #         if rank in ranks:
    #             _TENSOR_AND_DATA_PARALLEL_GROUP = group

    # Build the tensor + expert parallel groups
    # global _TENSOR_AND_EXPERT_PARALLEL_GROUP
    # assert (
    #        _TENSOR_AND_EXPERT_PARALLEL_GROUP is None
    # ), 'Tensor + expert parallel group is already initialized'
    # global _DATA_MODULO_EXPERT_PARALLEL_GROUP
    # assert (
    #        _DATA_MODULO_EXPERT_PARALLEL_GROUP is None
    # ), 'Data modulo expert group is already initialized'
    # tensor_and_data_group_size: int = tensor_model_parallel_size * data_parallel_size
    # num_tensor_and_data_groups: int = world_size // tensor_and_data_group_size
    # tensor_and_expert_group_size: int = tensor_model_parallel_size * expert_model_parallel_size
    # num_expert_groups: int = data_parallel_size // expert_model_parallel_size
    # for i in range(num_tensor_and_data_groups):
    #     for j in range(num_expert_groups):
    #         start_rank = i * tensor_and_data_group_size + j * tensor_and_expert_group_size
    #         end_rank = i * tensor_and_data_group_size + (j + 1) * tensor_and_expert_group_size
    #         ranks = range(start_rank, end_rank)
    #         group = torch.distributed.new_group(
    #             ranks, pg_options=get_nccl_options('tp_exp', nccl_comm_cfgs)
    #         )
    #         if rank in ranks:
    #             _TENSOR_AND_EXPERT_PARALLEL_GROUP = group
    #
    # for i in range(num_tensor_and_data_groups):
    #     start_rank = i * tensor_and_data_group_size
    #     end_rank = (i + 1) * tensor_and_data_group_size
    #     for j in range(tensor_and_expert_group_size):
    #         ranks = range(start_rank + j, end_rank, tensor_and_expert_group_size)
    #         group = torch.distributed.new_group(
    #             ranks, pg_options=get_nccl_options('dp_modulo_exp', nccl_comm_cfgs)
    #         )
    #         if rank in ranks:
    #             _DATA_MODULO_EXPERT_PARALLEL_GROUP = group

    # Initialize global memory buffer
    # This isn't really "parallel state" but there isn't another good place to
    # put this. If we end up with a more generic initialization of megatron-core
    # we could stick it there
    _set_global_memory_buffer()


def initialize_model_parallel_origin(
        tensor_model_parallel_size: int = 1,
        pipeline_model_parallel_size: int = 1,
        virtual_pipeline_model_parallel_size: Optional[int] = None,
        pipeline_model_parallel_split_rank: Optional[int] = None,
        use_sharp: bool = False,
        context_parallel_size: int = 1,
        expert_model_parallel_size: int = 1,
        nccl_communicator_config_path: Optional[str] = None,
) -> None:
    """Initialize model data parallel groups.

    Arguments:
        tensor_model_parallel_size (int, default = 1):
            The number of GPUs to split individual tensors across.

        pipeline_model_parallel_size (int, default = 1):
            The number of tensor parallel GPU groups to split the
            Transformer layers across. For example, if
            tensor_model_parallel_size is 4 and
            pipeline_model_parallel_size is 2, the model will be split
            into 2 groups of 4 GPUs.

        virtual_pipeline_model_parallel_size (int, optional):
            The number of stages that each pipeline group will have,
            interleaving as necessary. If None, no interleaving is
            performed. For example, if tensor_model_parallel_size is 1,
            pipeline_model_parallel_size is 4,
            virtual_pipeline_model_parallel_size is 2, and there are
            16 transformer layers in the model, the model will be
            split into 8 stages with two layers each and each GPU
            would get 2 stages as such (layer number starting with 1):

            GPU 0: [1, 2] [9, 10]
            GPU 1: [3, 4] [11, 12]
            GPU 2: [5, 6] [13, 14]
            GPU 3: [7, 8] [15, 16]

        pipeline_model_parallel_split_rank (int, optional):
            For models with both an encoder and decoder, the rank in
            pipeline to switch between encoder and decoder (i.e. the
            first rank of the decoder). This allows the user to set
            the pipeline parallel size of the encoder and decoder
            independently. For example, if
            pipeline_model_parallel_size is 8 and
            pipeline_model_parallel_split_rank is 3, then ranks 0-2
            will be the encoder and ranks 3-7 will be the decoder.

        use_sharp (bool, default = False):
            Set the use of SHARP for the collective communications of
            data-parallel process groups. When `True`, run barrier
            within each data-parallel process group, which specifies
            the SHARP application target groups.

        context_parallel_size (int, default = 1):
            The number of tensor parallel GPU groups to split the
            network input sequence length across. Compute of attention
            module requires tokens of full sequence length, so GPUs
            in a context parallel group need to communicate with each
            other to exchange information of other sequence chunks.
            Each GPU and its counterparts in other tensor parallel
            groups compose a context parallel group.

            For example, assume we have 8 GPUs, if tensor model parallel
            size is 4 and context parallel size is 2, the network input
            will be split into two sequence chunks, which are processed
            by 2 different groups of 4 GPUs. One chunk is processed by
            GPU0-3, the other chunk is processed by GPU4-7. Four groups
            are build to do context parallel communications: [GPU0, GPU4],
            [GPU1, GPU5], [GPU2, GPU6], and [GPU3, GPU7].

            Context parallelism partitions sequence length, so it has no
            impact on weights, which means weights are duplicated among
            GPUs in a context parallel group. Hence, weight gradients
            all-reduce is required in backward. For simplicity, we piggyback
            GPUs of context parallelism on data parallel group for
            weight gradient all-reduce.

        nccl_communicator_config_path (str, default = None):
            Path to the yaml file of NCCL communicator configurations.
            `min_ctas`, `max_ctas`, and `cga_cluster_size` can be set
            for each communicator.

    Let's say we have a total of 16 GPUs denoted by g0 ... g15 and we
    use 2 GPUs to parallelize the model tensor, and 4 GPUs to parallelize
    the model pipeline. The present function will
    create 8 tensor model-parallel groups, 4 pipeline model-parallel groups
    and 8 data-parallel groups as:
        8 data_parallel groups:
            [g0, g2], [g1, g3], [g4, g6], [g5, g7], [g8, g10], [g9, g11], [g12, g14], [g13, g15]
        8 tensor model-parallel groups:
            [g0, g1], [g2, g3], [g4, g5], [g6, g7], [g8, g9], [g10, g11], [g12, g13], [g14, g15]
        4 pipeline model-parallel groups:
            [g0, g4, g8, g12], [g1, g5, g9, g13], [g2, g6, g10, g14], [g3, g7, g11, g15]
    Note that for efficiency, the caller should make sure adjacent ranks
    are on the same DGX box. For example if we are using 2 DGX-1 boxes
    with a total of 16 GPUs, rank 0 to 7 belong to the first box and
    ranks 8 to 15 belong to the second box.

    """
    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized()
    world_size: int = torch.distributed.get_world_size()

    # 取消world_size的检查
    if (
            world_size
            % (tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size)
            != 0
    ):
        raise RuntimeError(
            f"world_size ({world_size}) is not divisible by tensor_model_parallel_size "
            f"({tensor_model_parallel_size}) x pipeline_model_parallel_size ({pipeline_model_parallel_size}) "
            f"x context_parallel_size ({context_parallel_size})"
        )

    # 取消data_parallel_size的计算，现由算法直接生成
    data_parallel_size: int = world_size // (
            tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size
    )

    if data_parallel_size % expert_model_parallel_size != 0:
        raise RuntimeError(
            f"data_parallel_size ({data_parallel_size}) is not divisible by expert_model_parallel_size "
        )

    if expert_model_parallel_size > 1 and context_parallel_size > 1:
        raise RuntimeError(
            f"combination of expert model prallellism and context parallelism is not supported"
        )

    # groups数现由算法直接生成
    num_tensor_model_parallel_groups: int = world_size // tensor_model_parallel_size
    num_pipeline_model_parallel_groups: int = world_size // pipeline_model_parallel_size

    if virtual_pipeline_model_parallel_size is not None:
        if not pipeline_model_parallel_size > 2:
            raise RuntimeError(
                "pipeline-model-parallel size should be greater than 2 with interleaved schedule"
            )
        global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
        global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
        _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = 0
        _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = virtual_pipeline_model_parallel_size

    if pipeline_model_parallel_split_rank is not None:
        global _PIPELINE_MODEL_PARALLEL_SPLIT_RANK
        _PIPELINE_MODEL_PARALLEL_SPLIT_RANK = pipeline_model_parallel_split_rank

    # 当前rank的全局号
    rank = torch.distributed.get_rank()

    nccl_comm_cfgs = {}
    if nccl_communicator_config_path is not None:
        try:
            import yaml
        except ImportError:
            raise RuntimeError(
                "Cannot import `yaml`. Setting custom nccl communicator configs "
                "requires the yaml package."
            )

        with open(nccl_communicator_config_path, "r") as stream:
            nccl_comm_cfgs = yaml.safe_load(stream)

    # Build the data-parallel groups.
    global _DATA_PARALLEL_GROUP
    global _DATA_PARALLEL_GROUP_GLOO
    global _DATA_PARALLEL_GLOBAL_RANKS
    global _DATA_PARALLEL_GROUP_WITH_CP
    global _DATA_PARALLEL_GROUP_WITH_CP_GLOO
    global _DATA_PARALLEL_GLOBAL_RANKS_WITH_CP
    assert _DATA_PARALLEL_GROUP is None, 'data parallel group is already initialized'
    all_data_parallel_group_ranks_with_cp = []
    for i in range(pipeline_model_parallel_size):
        start_rank = i * num_pipeline_model_parallel_groups
        end_rank = (i + 1) * num_pipeline_model_parallel_groups
        for j in range(context_parallel_size * tensor_model_parallel_size):
            ranks = range(
                start_rank + j, end_rank, context_parallel_size * tensor_model_parallel_size
            )
            group = torch.distributed.new_group(
                ranks, pg_options=get_nccl_options('dp', nccl_comm_cfgs)
            )
            group_gloo = torch.distributed.new_group(ranks, backend="gloo")
            if rank in ranks:
                _DATA_PARALLEL_GROUP = group
                _DATA_PARALLEL_GROUP_GLOO = group_gloo
                _DATA_PARALLEL_GLOBAL_RANKS = ranks
        for j in range(tensor_model_parallel_size):
            ranks_with_cp = range(start_rank + j, end_rank, tensor_model_parallel_size)
            all_data_parallel_group_ranks_with_cp.append(list(ranks_with_cp))
            group_with_cp = torch.distributed.new_group(
                ranks_with_cp, pg_options=get_nccl_options('dp_cp', nccl_comm_cfgs)
            )
            group_with_cp_gloo = torch.distributed.new_group(ranks_with_cp, backend="gloo")
            if rank in ranks_with_cp:
                _DATA_PARALLEL_GROUP_WITH_CP = group_with_cp
                _DATA_PARALLEL_GROUP_WITH_CP_GLOO = group_with_cp_gloo
                _DATA_PARALLEL_GLOBAL_RANKS_WITH_CP = ranks_with_cp

    # Apply SHARP to DP process groups
    if use_sharp:
        if rank == 0:
            print(
                "The number of process groups to use SHARP with depends on the type "
                "of the network switch. Nvidia QM1 switch supports SAHRP up to 8 "
                "process groups and QM2 supports up to 256 process groups. We apply "
                "SHARP to the communications of the data-parallel domain. If the "
                "number of data-parallel process groups is larger than the max "
                "process groups that the network switch supports, the communication "
                "will fall back to non-SHARP operators. To enable SHARP, "
                "`#SBATCH_NETWORK=sharp` should be set in the sbatch script."
            )
        torch.distributed.barrier(
            group=get_data_parallel_group(with_context_parallel=context_parallel_size > 1),
            device_ids=[torch.cuda.current_device()],
        )
        # Set `NCCL_SHARP_DISABLE=1` to restrict SHARP application to DP process groups
        os.environ["NCCL_SHARP_DISABLE"] = "1"

    # Build the context-parallel groups.
    global _CONTEXT_PARALLEL_GROUP
    global _CONTEXT_PARALLEL_GLOBAL_RANKS
    assert _CONTEXT_PARALLEL_GROUP is None, 'context parallel group is already initialized'
    for i in range(pipeline_model_parallel_size):
        for j in range(data_parallel_size):
            start_rank = (
                    i * num_pipeline_model_parallel_groups
                    + j * tensor_model_parallel_size * context_parallel_size
            )
            end_rank = (
                    i * num_pipeline_model_parallel_groups
                    + (j + 1) * tensor_model_parallel_size * context_parallel_size
            )
            for k in range(tensor_model_parallel_size):
                ranks = range(start_rank + k, end_rank, tensor_model_parallel_size)
                group = torch.distributed.new_group(
                    ranks, pg_options=get_nccl_options('cp', nccl_comm_cfgs)
                )
                if rank in ranks:
                    _CONTEXT_PARALLEL_GROUP = group
                    _CONTEXT_PARALLEL_GLOBAL_RANKS = ranks

    # Build the model-parallel groups.
    global _MODEL_PARALLEL_GROUP
    assert _MODEL_PARALLEL_GROUP is None, 'model parallel group is already initialized'
    for i in range(data_parallel_size * context_parallel_size):
        ranks = [
            data_parallel_group_ranks_with_cp[i]
            for data_parallel_group_ranks_with_cp in all_data_parallel_group_ranks_with_cp
        ]
        group = torch.distributed.new_group(
            ranks, pg_options=get_nccl_options('mp', nccl_comm_cfgs)
        )
        if rank in ranks:
            _MODEL_PARALLEL_GROUP = group

    # Build the tensor model-parallel groups.
    global _TENSOR_MODEL_PARALLEL_GROUP
    assert (
            _TENSOR_MODEL_PARALLEL_GROUP is None
    ), 'tensor model parallel group is already initialized'
    for i in range(num_tensor_model_parallel_groups):
        ranks = range(i * tensor_model_parallel_size, (i + 1) * tensor_model_parallel_size)
        group = torch.distributed.new_group(
            ranks, pg_options=get_nccl_options('tp', nccl_comm_cfgs)
        )
        if rank in ranks:
            _TENSOR_MODEL_PARALLEL_GROUP = group

    # Build the pipeline model-parallel groups and embedding groups
    # (first and last rank in each pipeline model-parallel group).
    global _PIPELINE_MODEL_PARALLEL_GROUP
    global _PIPELINE_GLOBAL_RANKS
    assert (
            _PIPELINE_MODEL_PARALLEL_GROUP is None
    ), 'pipeline model parallel group is already initialized'
    global _EMBEDDING_GROUP
    global _EMBEDDING_GLOBAL_RANKS
    assert _EMBEDDING_GROUP is None, 'embedding group is already initialized'
    global _POSITION_EMBEDDING_GROUP
    global _POSITION_EMBEDDING_GLOBAL_RANKS
    assert _POSITION_EMBEDDING_GROUP is None, 'position embedding group is already initialized'
    for i in range(num_pipeline_model_parallel_groups):
        ranks = range(i, world_size, num_pipeline_model_parallel_groups)
        group = torch.distributed.new_group(
            ranks, pg_options=get_nccl_options('pp', nccl_comm_cfgs)
        )
        if rank in ranks:
            _PIPELINE_MODEL_PARALLEL_GROUP = group
            _PIPELINE_GLOBAL_RANKS = ranks
        # Setup embedding group (to exchange gradients between
        # first and last stages).
        if len(ranks) > 1:
            embedding_ranks = [ranks[0], ranks[-1]]
            position_embedding_ranks = [ranks[0]]
            if pipeline_model_parallel_split_rank is not None:
                if ranks[pipeline_model_parallel_split_rank] not in embedding_ranks:
                    embedding_ranks = [
                        ranks[0],
                        ranks[pipeline_model_parallel_split_rank],
                        ranks[-1],
                    ]
                if ranks[pipeline_model_parallel_split_rank] not in position_embedding_ranks:
                    position_embedding_ranks = [ranks[0], ranks[pipeline_model_parallel_split_rank]]
        else:
            embedding_ranks = ranks
            position_embedding_ranks = ranks

        group = torch.distributed.new_group(
            embedding_ranks, pg_options=get_nccl_options('embd', nccl_comm_cfgs)
        )
        if rank in embedding_ranks:
            _EMBEDDING_GROUP = group
        if rank in ranks:
            _EMBEDDING_GLOBAL_RANKS = embedding_ranks

        group = torch.distributed.new_group(
            position_embedding_ranks, pg_options=get_nccl_options('embd', nccl_comm_cfgs)
        )
        if rank in position_embedding_ranks:
            _POSITION_EMBEDDING_GROUP = group
        if rank in ranks:
            _POSITION_EMBEDDING_GLOBAL_RANKS = position_embedding_ranks

    # Build the tensor + data parallel groups.
    global _TENSOR_AND_DATA_PARALLEL_GROUP
    global _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP
    assert (
            _TENSOR_AND_DATA_PARALLEL_GROUP is None
    ), 'Tensor + data parallel group is already initialized'
    tensor_and_data_group_size_with_cp: int = tensor_model_parallel_size * data_parallel_size * context_parallel_size
    num_tensor_and_data_groups_with_cp: int = world_size // tensor_and_data_group_size_with_cp
    for i in range(num_tensor_and_data_groups_with_cp):
        start_rank = i * tensor_and_data_group_size_with_cp
        end_rank = start_rank + tensor_and_data_group_size_with_cp
        ranks = range(start_rank, end_rank)
        group = torch.distributed.new_group(
            ranks, pg_options=get_nccl_options('tp_dp_cp', nccl_comm_cfgs)
        )
        if rank in ranks:
            _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP = group

        for j in range(context_parallel_size):
            ranks = []
            for k in range(data_parallel_size):
                start_rank = (
                        i * tensor_and_data_group_size_with_cp
                        + j * tensor_model_parallel_size
                        + k * tensor_model_parallel_size * context_parallel_size
                )
                end_rank = start_rank + tensor_model_parallel_size
                ranks = ranks + list(range(start_rank, end_rank))
            group = torch.distributed.new_group(
                ranks, pg_options=get_nccl_options('tp_dp', nccl_comm_cfgs)
            )
            if rank in ranks:
                _TENSOR_AND_DATA_PARALLEL_GROUP = group

    # Build the tensor + expert parallel groups
    global _TENSOR_AND_EXPERT_PARALLEL_GROUP
    assert (
            _TENSOR_AND_EXPERT_PARALLEL_GROUP is None
    ), 'Tensor + expert parallel group is already initialized'
    global _DATA_MODULO_EXPERT_PARALLEL_GROUP
    assert (
            _DATA_MODULO_EXPERT_PARALLEL_GROUP is None
    ), 'Data modulo expert group is already initialized'
    tensor_and_data_group_size: int = tensor_model_parallel_size * data_parallel_size
    num_tensor_and_data_groups: int = world_size // tensor_and_data_group_size
    tensor_and_expert_group_size: int = tensor_model_parallel_size * expert_model_parallel_size
    num_expert_groups: int = data_parallel_size // expert_model_parallel_size
    for i in range(num_tensor_and_data_groups):
        for j in range(num_expert_groups):
            start_rank = i * tensor_and_data_group_size + j * tensor_and_expert_group_size
            end_rank = i * tensor_and_data_group_size + (j + 1) * tensor_and_expert_group_size
            ranks = range(start_rank, end_rank)
            group = torch.distributed.new_group(
                ranks, pg_options=get_nccl_options('tp_exp', nccl_comm_cfgs)
            )
            if rank in ranks:
                _TENSOR_AND_EXPERT_PARALLEL_GROUP = group

    for i in range(num_tensor_and_data_groups):
        start_rank = i * tensor_and_data_group_size
        end_rank = (i + 1) * tensor_and_data_group_size
        for j in range(tensor_and_expert_group_size):
            ranks = range(start_rank + j, end_rank, tensor_and_expert_group_size)
            group = torch.distributed.new_group(
                ranks, pg_options=get_nccl_options('dp_modulo_exp', nccl_comm_cfgs)
            )
            if rank in ranks:
                _DATA_MODULO_EXPERT_PARALLEL_GROUP = group

    # Initialize global memory buffer
    # This isn't really "parallel state" but there isn't another good place to
    # put this. If we end up with a more generic initialization of megatron-core
    # we could stick it there
    _set_global_memory_buffer()


def is_unitialized():
    """Useful for code segments that may be accessed with or without mpu initialization"""
    return _DATA_PARALLEL_GROUP is None


def model_parallel_is_initialized():
    """Check if model and data parallel groups are initialized."""
    if (
            _TENSOR_MODEL_PARALLEL_GROUP is None
            or _PIPELINE_MODEL_PARALLEL_GROUP is None
            or _DATA_PARALLEL_GROUP is None
    ):
        return False
    return True


def get_model_parallel_group():
    """Get the model parallel group the caller rank belongs to."""
    # 返回所在的模型并行组
    assert _MODEL_PARALLEL_GROUP is not None, 'model parallel group is not initialized'
    return _MODEL_PARALLEL_GROUP


def get_tensor_model_parallel_group(check_initialized=True):
    """Get the tensor model parallel group the caller rank belongs to."""
    # 返回所在的tensor并行组
    if check_initialized:
        assert (
                _TENSOR_MODEL_PARALLEL_GROUP is not None
        ), 'tensor model parallel group is not initialized'
    return _TENSOR_MODEL_PARALLEL_GROUP


def get_pipeline_model_parallel_group():
    """Get the pipeline model parallel group the caller rank belongs to."""
    assert (
        _PIPELINE_MODEL_PARALLEL_GROUP
    ), 'pipeline_model parallel group is not initialized'

    return _PIPELINE_MODEL_PARALLEL_GROUP


def get_data_parallel_group(with_context_parallel=False):
    """Get the data parallel group the caller rank belongs to."""
    if with_context_parallel:
        assert (
                _DATA_PARALLEL_GROUP_WITH_CP is not None
        ), 'data parallel group with context parallel combined is not initialized'
        return _DATA_PARALLEL_GROUP_WITH_CP
    else:
        assert _DATA_PARALLEL_GROUP, 'data parallel group is not initialized'
        return _DATA_PARALLEL_GROUP


def get_data_parallel_group_gloo(with_context_parallel=False):
    """Get the data parallel group-gloo the caller rank belongs to."""
    if with_context_parallel:
        assert (
                _DATA_PARALLEL_GROUP_WITH_CP_GLOO is not None
        ), 'data parallel group-gloo with context parallel combined is not initialized'
        return _DATA_PARALLEL_GROUP_WITH_CP_GLOO
    else:
        assert _DATA_PARALLEL_GROUP_GLOO is not None, 'data parallel group-gloo is not initialized'
        return _DATA_PARALLEL_GROUP_GLOO


def get_context_parallel_group(check_initialized=True):
    """Get the context parallel group the caller rank belongs to."""
    if check_initialized:
        assert _CONTEXT_PARALLEL_GROUP is not None, 'context parallel group is not initialized'
    return _CONTEXT_PARALLEL_GROUP


def get_context_parallel_global_ranks(check_initialized=True):
    """Get all global ranks of the context parallel group that the caller rank belongs to."""
    if check_initialized:
        assert (
                _CONTEXT_PARALLEL_GLOBAL_RANKS is not None
        ), 'context parallel group is not initialized'
    return _CONTEXT_PARALLEL_GLOBAL_RANKS


def get_embedding_group():
    """Get the embedding group the caller rank belongs to."""
    assert _EMBEDDING_GROUP, 'embedding group is not initialized'
    return _EMBEDDING_GROUP


def get_embedding_global_ranks():
    return _EMBEDDING_GLOBAL_RANKS


def get_position_embedding_group():
    """Get the position embedding group the caller rank belongs to."""
    assert _POSITION_EMBEDDING_GROUP, 'position embedding group is not initialized'
    return _POSITION_EMBEDDING_GROUP


def get_amax_reduction_group(with_context_parallel=False):
    """Get the FP8 amax reduction group the caller rank belongs to."""
    if with_context_parallel:
        assert (
                _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP is not None
        ), 'FP8 amax reduction group is not initialized'
        return _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP
    else:
        assert (
                _TENSOR_AND_DATA_PARALLEL_GROUP is not None
        ), 'FP8 amax reduction group is not initialized'
        return _TENSOR_AND_DATA_PARALLEL_GROUP


def get_tensor_and_data_parallel_group(with_context_parallel=False):
    """Get the tensor and data parallel group the caller rank belongs to."""
    if with_context_parallel:
        assert (
                _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP is not None
        ), 'tensor and data parallel group is not initialized'
        return _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP
    else:
        assert (
                _TENSOR_AND_DATA_PARALLEL_GROUP is not None
        ), 'tensor and data parallel group is not initialized'
        return _TENSOR_AND_DATA_PARALLEL_GROUP


def get_tensor_and_expert_parallel_group():
    assert (
            _TENSOR_AND_EXPERT_PARALLEL_GROUP is not None
    ), 'tensor and expert parallel group is not initialized'
    return _TENSOR_AND_EXPERT_PARALLEL_GROUP


def get_data_modulo_expert_parallel_group():
    assert (
            _DATA_MODULO_EXPERT_PARALLEL_GROUP is not None
    ), 'data modulo expert parallel group is not initialized'
    return _DATA_MODULO_EXPERT_PARALLEL_GROUP


def set_tensor_model_parallel_world_size(world_size):
    """Set the tensor model parallel size"""
    global _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = world_size


def set_pipeline_model_parallel_world_size(world_size):
    """Set the pipeline model parallel size"""
    global _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = world_size


def set_virtual_pipeline_model_parallel_world_size(world_size):
    """Set the pipeline model parallel size"""
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = world_size


def get_tensor_model_parallel_world_size():
    """Return world size for the tensor model parallel group."""
    global _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    if _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE is not None:
        return _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    return torch.distributed.get_world_size(group=get_tensor_model_parallel_group())


def get_pipeline_model_parallel_world_size():
    """Return world size for the pipeline model parallel group."""
    global _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    if _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE is not None:
        return _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    pp_group = _PIPELINE_MODEL_PARALLEL_GROUP[0]
    return torch.distributed.get_world_size(group=pp_group)


def set_tensor_model_parallel_rank(rank):
    """Set tensor model parallel rank."""
    global _MPU_TENSOR_MODEL_PARALLEL_RANK
    _MPU_TENSOR_MODEL_PARALLEL_RANK = rank


def set_pipeline_model_parallel_rank(rank):
    """Set pipeline model parallel rank."""
    global _MPU_PIPELINE_MODEL_PARALLEL_RANK
    _MPU_PIPELINE_MODEL_PARALLEL_RANK = rank


def set_pipeline_model_parallel_split_rank(rank):
    """Set pipeline model parallel split rank."""
    global _PIPELINE_MODEL_PARALLEL_SPLIT_RANK
    _PIPELINE_MODEL_PARALLEL_SPLIT_RANK = rank


def get_tensor_model_parallel_rank():
    """Return my rank for the tensor model parallel group."""
    # 返回在特定进程组中的相对rank号，假如group = [1,2,3],那么rank1返回的是0
    global _MPU_TENSOR_MODEL_PARALLEL_RANK
    if _MPU_TENSOR_MODEL_PARALLEL_RANK is not None:
        return _MPU_TENSOR_MODEL_PARALLEL_RANK
    return torch.distributed.get_rank(group=get_tensor_model_parallel_group())


def get_pipeline_model_parallel_rank():
    """Return my rank for the pipeline model parallel group."""
    global _MPU_PIPELINE_MODEL_PARALLEL_RANK
    global _PIPELINE_GLOBAL_RANKS
    if _MPU_PIPELINE_MODEL_PARALLEL_RANK is not None:
        return _MPU_PIPELINE_MODEL_PARALLEL_RANK
    pp_group = _PIPELINE_MODEL_PARALLEL_GROUP[0]
    rank = torch.distributed.get_rank()
    # return _PIPELINE_GLOBAL_RANKS[0].index(rank)
    #print(f"当前rank是{torch.distributed.get_rank()},相对位置是{_PIPELINE_GLOBAL_RANKS[0].index(rank)}")
    return _PIPELINE_GLOBAL_RANKS[0].index(rank)
    # return torch.distributed.get_rank(group=pp_group)

def get_pipeline_model_parallel_split_rank():
    """Return pipeline model parallel split rank."""
    global _PIPELINE_MODEL_PARALLEL_SPLIT_RANK
    return _PIPELINE_MODEL_PARALLEL_SPLIT_RANK


def is_pipeline_first_stage(ignore_virtual=False):
    """Return True if in the first pipeline model-parallel stage, False otherwise."""
    if not ignore_virtual:
        if (
                get_virtual_pipeline_model_parallel_world_size() is not None
                and get_virtual_pipeline_model_parallel_rank() != 0
        ):
            return False
    return get_pipeline_model_parallel_rank() == 0


def is_pipeline_last_stage(ignore_virtual=False):
    """Return True if in the last pipeline model-parallel stage, False otherwise."""
    if not ignore_virtual:
        virtual_pipeline_model_parallel_world_size = (
            get_virtual_pipeline_model_parallel_world_size()
        )
        if virtual_pipeline_model_parallel_world_size is not None and get_virtual_pipeline_model_parallel_rank() != (
                virtual_pipeline_model_parallel_world_size - 1
        ):
            return False
    return get_pipeline_model_parallel_rank() == (get_pipeline_model_parallel_world_size() - 1)


def is_rank_in_embedding_group(ignore_virtual=False):
    """Return true if current rank is in embedding group, False otherwise."""
    rank = torch.distributed.get_rank()
    global _EMBEDDING_GLOBAL_RANKS
    # if ignore_virtual:
    #     return rank in _EMBEDDING_GLOBAL_RANKS
    for idx, value in _EMBEDDING_GLOBAL_RANKS.items():
        if rank in value:
            if rank == value[0]:
                return is_pipeline_first_stage(ignore_virtual=False)
            elif rank == value[-1]:
                return is_pipeline_last_stage(ignore_virtual=False)
            else:
                return True
        # if rank in _EMBEDDING_GLOBAL_RANKS:
        #     if rank == _EMBEDDING_GLOBAL_RANKS[0]:
        #         return is_pipeline_first_stage(ignore_virtual=False)
        #     elif rank == _EMBEDDING_GLOBAL_RANKS[-1]:
        #         return is_pipeline_last_stage(ignore_virtual=False)
        #     else:
        #         return True
    return False


def is_rank_in_position_embedding_group():
    """Return true if current rank is in position embedding group, False otherwise."""
    rank = torch.distributed.get_rank()
    global _POSITION_EMBEDDING_GLOBAL_RANKS
    return rank in _POSITION_EMBEDDING_GLOBAL_RANKS


def is_pipeline_stage_before_split(rank=None):
    """Return True if pipeline stage executes encoder block for a model
    with both encoder and decoder."""
    if get_pipeline_model_parallel_world_size() == 1:
        return True
    if rank is None:
        rank = get_pipeline_model_parallel_rank()
    global _PIPELINE_MODEL_PARALLEL_SPLIT_RANK
    if _PIPELINE_MODEL_PARALLEL_SPLIT_RANK is None:
        return True
    if rank < _PIPELINE_MODEL_PARALLEL_SPLIT_RANK:
        return True
    return False


def is_pipeline_stage_after_split(rank=None):
    """Return True if pipeline stage executes decoder block for a model
    with both encoder and decoder."""
    if get_pipeline_model_parallel_world_size() == 1:
        return True
    if rank is None:
        rank = get_pipeline_model_parallel_rank()
    global _PIPELINE_MODEL_PARALLEL_SPLIT_RANK
    if _PIPELINE_MODEL_PARALLEL_SPLIT_RANK is None:
        return True
    if rank >= _PIPELINE_MODEL_PARALLEL_SPLIT_RANK:
        return True
    return False


def is_pipeline_stage_at_split():
    """Return true if pipeline stage executes decoder block and next
    stage executes encoder block for a model with both encoder and
    decoder."""
    rank = get_pipeline_model_parallel_rank()
    return is_pipeline_stage_before_split(rank) and is_pipeline_stage_after_split(rank + 1)


def get_virtual_pipeline_model_parallel_rank():
    """Return the virtual pipeline-parallel rank."""
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
    return _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK


def set_virtual_pipeline_model_parallel_rank(rank):
    """Set the virtual pipeline-parallel rank."""
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
    _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = rank


def get_virtual_pipeline_model_parallel_world_size():
    """Return the virtual pipeline-parallel world size."""
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    return _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE


def get_tensor_model_parallel_src_rank():
    """Calculate the global rank corresponding to the first local rank
    in the tensor model parallel group."""
    global _TENSOR_PARALLEL_RANKS
    assert _TENSOR_PARALLEL_RANKS is not None, "Tensor parallel group is not initialized"
    # global_rank = torch.distributed.get_rank()
    # local_world_size = get_tensor_model_parallel_world_size()
    return _TENSOR_PARALLEL_RANKS[0]


def get_data_parallel_src_rank(with_context_parallel=False):
    """Calculate the global rank corresponding to the first local rank
    in the data parallel group."""
    # DP参数随机初始化的时候才会用到
    if with_context_parallel:
        assert (
                _DATA_PARALLEL_GLOBAL_RANKS_WITH_CP is not None
        ), "Data parallel group with context parallel combined is not initialized"
        return _DATA_PARALLEL_GLOBAL_RANKS_WITH_CP[0]
    else:
        assert _DATA_PARALLEL_GLOBAL_RANKS is not None, "Data parallel group is not initialized"
        return _DATA_PARALLEL_GLOBAL_RANKS[0]


def get_pipeline_model_parallel_first_rank():
    """Return the global rank of the first process in the pipeline for the
    current tensor parallel group"""
    assert _PIPELINE_GLOBAL_RANKS, "Pipeline parallel group is not initialized"
    return _PIPELINE_GLOBAL_RANKS[0][0]


def get_pipeline_model_parallel_last_rank():
    """Return the global rank of the last process in the pipeline for the
    current tensor parallel group"""
    assert _PIPELINE_GLOBAL_RANKS, "Pipeline parallel group is not initialized"
    last_rank_local = get_pipeline_model_parallel_world_size() - 1
    return _PIPELINE_GLOBAL_RANKS[0][last_rank_local]


def get_pipeline_model_parallel_next_rank():
    """Return the global rank that follows the caller in the pipeline"""
    assert _PIPELINE_GLOBAL_RANKS, "Pipeline parallel group is not initialized"
    rank_in_pipeline = get_pipeline_model_parallel_rank()
    #print(f"当前rank为{torch.distributed.get_rank()}rank_in_pipeline为{rank_in_pipeline}")
    world_size = get_pipeline_model_parallel_world_size()
    rank_next = []
    for idx, value in _PIPELINE_GLOBAL_RANKS.items():
        #print(f"当前rank为{torch.distributed.get_rank()}value为{value}")
        rank_next.append(value[(rank_in_pipeline + 1) % world_size])
    return rank_next


def get_pipeline_model_parallel_prev_rank():
    """Return the global rank that preceeds the caller in the pipeline"""
    assert _PIPELINE_GLOBAL_RANKS, "Pipeline parallel group is not initialized"
    rank_in_pipeline = get_pipeline_model_parallel_rank()
    world_size = get_pipeline_model_parallel_world_size()
    rank_prev = []
    for idx, value in _PIPELINE_GLOBAL_RANKS.items():
        rank_prev.append(value[(rank_in_pipeline - 1) % world_size])
    return rank_prev


def get_data_parallel_world_size(with_context_parallel=False):
    """Return world size for the data parallel group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size(
            group=get_data_parallel_group(with_context_parallel=with_context_parallel)[0]
        )
    else:
        return 0


def get_data_parallel_rank(with_context_parallel=False):
    """Return my rank for the data parallel group."""
    global _DATA_PARALLEL_GROUP
    group = _DATA_PARALLEL_GROUP[0]
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(
            group=group
        )
    else:
        return 0


def get_context_parallel_world_size():
    """Return world size for the context parallel group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size(group=get_context_parallel_group())
    else:
        return 0


def get_context_parallel_rank():
    """Return my rank for the context parallel group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(group=get_context_parallel_group())
    else:
        return 0


def get_expert_model_parallel_world_size():
    """Return world size for the expert model parallel group"""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        tensor_and_expert_parallel_world_size = torch.distributed.get_world_size(
            group=get_tensor_and_expert_parallel_group()
        )
        return tensor_and_expert_parallel_world_size // get_tensor_model_parallel_world_size()
    else:
        return 0


def get_tensor_and_expert_parallel_world_size():
    """Return world size for the expert model parallel group times model parallel group.
       Currently, each expert will also be distributed across TP group by default.
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        tensor_and_expert_parallel_world_size = torch.distributed.get_world_size(
            group=get_tensor_and_expert_parallel_group()
        )
        return tensor_and_expert_parallel_world_size
    else:
        return 0


def get_expert_model_parallel_rank():
    """Return my rank for the expert parallel group"""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        tensor_and_expert_parallel_rank = torch.distributed.get_rank(
            group=get_tensor_and_expert_parallel_group()
        )
        return tensor_and_expert_parallel_rank // get_tensor_model_parallel_world_size()
    else:
        return 0


def get_data_modulo_expert_parallel_rank():
    """Return my rank for the context parallel group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(group=get_data_modulo_expert_parallel_group())
    else:
        return 0


def _set_global_memory_buffer():
    """Initialize global buffer"""
    global _GLOBAL_MEMORY_BUFFER
    assert _GLOBAL_MEMORY_BUFFER is None, 'global memory buffer is already initialized'
    _GLOBAL_MEMORY_BUFFER = GlobalMemoryBuffer()


def get_global_memory_buffer():
    """Return the global GlobalMemoryBuffer object"""
    assert _GLOBAL_MEMORY_BUFFER is not None, 'global memory buffer is not initialized'
    return _GLOBAL_MEMORY_BUFFER


def destroy_global_memory_buffer():
    """Sets the global memory buffer to None"""
    global _GLOBAL_MEMORY_BUFFER
    _GLOBAL_MEMORY_BUFFER = None


def get_layer_rank_mapping():
    """Return layer rank mapping"""
    with open("dp_layer_rank_mapping.json", 'r') as f:
        layer_rank_mapping = json.load(f)
    return layer_rank_mapping


def find_matching_layer_dp_group():
    args = get_args()
    layer_rank_mapping = get_layer_rank_mapping()
    num_layer = args.num_layers
    # dp_group = list(layer_rank_mapping.keys())
    matching_layer = {}
    for i in range(1, num_layer + 1):
        matching_layer[i] = []
    layer_curr = 1
    for dp_num, rank_layer in layer_rank_mapping.items():
        for rank, layer in rank_layer.items():
            while True:
                if layer_curr <= layer[1]:
                    matching_layer[layer_curr].append(rank)
                    layer_curr += 1
                else:
                    break
        layer_curr = 1
    print(f"matching_layer为{matching_layer}====================================")
    return matching_layer


def get_data_parallel_global_ranks():
    """Return all global ranks of the data parallel group that the caller rank belongs to."""
    assert _DATA_PARALLEL_GLOBAL_RANKS, 'data parallel group is not initialized'
    return _DATA_PARALLEL_GLOBAL_RANKS


def destroy_model_parallel():
    """Set the groups to none."""
    global _MODEL_PARALLEL_GROUP
    _MODEL_PARALLEL_GROUP = None
    global _TENSOR_MODEL_PARALLEL_GROUP
    _TENSOR_MODEL_PARALLEL_GROUP = None
    global _PIPELINE_MODEL_PARALLEL_GROUP
    _PIPELINE_MODEL_PARALLEL_GROUP = None
    global _DATA_PARALLEL_GROUP
    _DATA_PARALLEL_GROUP = None
    global _DATA_PARALLEL_GROUP_WITH_CP
    _DATA_PARALLEL_GROUP_WITH_CP = None
    global _CONTEXT_PARALLEL_GROUP
    _CONTEXT_PARALLEL_GROUP = None
    global _CONTEXT_PARALLEL_GLOBAL_RANKS
    _CONTEXT_PARALLEL_GLOBAL_RANKS = None
    global _EMBEDDING_GROUP
    _EMBEDDING_GROUP = None
    global _POSITION_EMBEDDING_GROUP
    _POSITION_EMBEDDING_GROUP = None
    global _TENSOR_AND_DATA_PARALLEL_GROUP
    _TENSOR_AND_DATA_PARALLEL_GROUP = None
    global _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP
    _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP = None
    global _TENSOR_AND_EXPERT_PARALLEL_GROUP
    _TENSOR_AND_EXPERT_PARALLEL_GROUP = None
    global _DATA_MODULO_EXPERT_PARALLEL_GROUP
    _DATA_MODULO_EXPERT_PARALLEL_GROUP = None
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
    _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = None
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
    global _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = None
    global _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
    global _MPU_TENSOR_MODEL_PARALLEL_RANK
    _MPU_TENSOR_MODEL_PARALLEL_RANK = None
    global _MPU_PIPELINE_MODEL_PARALLEL_RANK
    _MPU_PIPELINE_MODEL_PARALLEL_RANK = None
    global _GLOBAL_MEMORY_BUFFER
    _GLOBAL_MEMORY_BUFFER = None


def contain_only_one_sublist(list):
    if len(list) != 1:
        return False
    if not isinstance(list[0], list):
        return False
    return True


def contain_only_one_element(list):
    if len(list) != 1:
        return False
    if not isinstance(list[0], int):
        return False
    return True


def change_to_symmetric_list(list1):
    list_var = [[item] if not isinstance(item, list) else item for item in list1]
    maxlength = 1
    for value in list_var:
        if len(value) >= maxlength:
            maxlength = len(value)
    if maxlength > 1:
        for value in list_var:
            if len(value) <= 1:
                value.append(value[0])
    return list_var


def is_tensor_parallel():
    global _TENSOR_PARALLEL_RANKS
    if len(_TENSOR_PARALLEL_RANKS) == 1:
        return False
    return True


def print_memory_usage():
    print(f"当前的rank为:{torch.distributed.get_rank()}")
    print(f"当前占用的显存为{torch.cuda.memory_allocated() / 1024 / 1024}MB")
    print(f"当前占用的显存为{torch.cuda.memory_reserved() / 1024 / 1024}MB")


def get_data_parallel_global_layers():
    global _DATA_PARALLEL_GLOBAL_LAYERS
    return _DATA_PARALLEL_GLOBAL_LAYERS


def get_data_parallel_num_layer():
    global _DATA_PARALLEL_NUM_LAYER
    return int(_DATA_PARALLEL_NUM_LAYER)


def get_data_parallel_offset():
    global _DATA_PARALLEL_OFFSETS
    return int(_DATA_PARALLEL_OFFSETS)
