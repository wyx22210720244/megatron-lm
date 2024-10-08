# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import math
from logging import getLogger
from typing import Dict, List

import torch

from .. import parallel_state
from megatron import get_args

logger = getLogger(__name__)


def shard_buffer(buffer: torch.Tensor, data_parallel_world_size: int):
    """
    Shard buffer into data_parallel_world_size chunks of equal size.
    """
    assert buffer.numel() % data_parallel_world_size == 0
    shard_size = buffer.numel() // data_parallel_world_size
    sharded_buffer = [
        buffer[(r * shard_size): ((r + 1) * shard_size)] for r in range(data_parallel_world_size)
    ]
    return sharded_buffer


class Bucket:
    """
    Bucket to keep track of a subset of the model's gradients. Provides functionality to register
    when params in the bucket have grads ready to be synced; an asynchronous communication call
    is automatically launched when _all_ params in the bucket have grads ready.

    Arguments:
        params: List of parameters whose gradients are collated in this bucket.
        data: View in larger GradBuffer that this bucket is responsible for.
        offset: Offset of this bucket's view in the larger GradBuffer.
        data_parallel_group: Data-parallel process group.
        data_parallel_world_size: World size using the data-parallel group group.
        overlap_grad_reduce: If true, overlap communication with backprop computation by
            breaking up grads into buckets. If false, single synchronous communication call
            is used instead.
        use_distributed_optimizer: If true, issue reduce-scatter communication calls as part
            of distributed optimizer. If false, issue all-reduce communication calls.
    """

    def __init__(
            self,
            params: List[torch.nn.Parameter],
            data: torch.Tensor,
            offset: int,
            data_parallel_group: torch.distributed.ProcessGroup,
            data_parallel_world_size: int,
            overlap_grad_reduce: bool,
            use_distributed_optimizer: bool,
            param_numel_count,
            layer_weight_numel_count,
    ):
        # State for bookkeeping: params is the set of parameters this bucket is
        # responsible for, params_with_grad is the set of parameters with grads
        # available. When overlap_grad_reduce is True, communication (all-reduce
        # or reduce-scatter) is issued when params_with_grad equals params.
        self.param_numel_count = param_numel_count
        self.layer_weight_numel_count = layer_weight_numel_count
        self.params_list = params
        self.params = set(params)
        self.params_with_grad = set()
        self.data = data
        # The distributed optimizer needs to keep track of this bucket's offset
        # within the full grad_buffer.
        self.offset = offset
        self.data_parallel_group = data_parallel_group
        self.data_parallel_world_size = data_parallel_world_size
        self.data_parallel_rank = torch.distributed.get_rank(group=data_parallel_group)
        self.overlap_grad_reduce = overlap_grad_reduce
        self.use_distributed_optimizer = use_distributed_optimizer

        self.reset()

    def reset(self):
        """
        Reset metadata in bucket in preparation for the next iteration of training.
        """
        self.params_with_grad = set()
        self.communication_handle = None
        self.communication_issued = False

    def start_grad_sync(self):
        """
        Initiates grad sync (all-reduce or reduce-scatter) communication operation
        for this bucket.

        When overlap_grad_reduce is set to True, dispatches an asynchronous
        communication call. When overlap_grad_reduce is set to False, makes
        synchronous call.
        """
        assert (
                self.communication_handle is None and not self.communication_issued
        ), 'Should not have multiple communication calls in flight at once'
        rank = torch.distributed.get_rank()
        self.data /= self.data_parallel_world_size
        # print("当前rank是{torch.distributed.get_rank()}开始all-reduce++++++++++++++++++++++++++++++++++")
        data_parallel_group = parallel_state.get_data_parallel_group()
        data_parallel_layer = parallel_state.get_data_parallel_global_layers()
        start_index = 0
        if parallel_state.is_pipeline_first_stage():
            for idx, group in data_parallel_group.items():
                if idx == 0:
                    end_index = (self.param_numel_count["embedding_weight"] + self.param_numel_count["layer_weight"] *
                                 data_parallel_layer[idx])
                    self.communication_handle = torch.distributed.all_reduce(
                        self.data[start_index:end_index], group=group, async_op=self.overlap_grad_reduce
                    )
                    start_index = end_index
                else:
                    end_index = start_index + self.param_numel_count["layer_weight"] * data_parallel_layer[idx]
                    self.communication_handle = torch.distributed.all_reduce(
                        self.data[start_index:end_index], group=group, async_op=self.overlap_grad_reduce
                    )
                    start_index = end_index
                    # 当DP组中只有tensor并行时
                    # 该组中没有final—word-embedding
                    # 与其他组all-reduce时需要补全
                    if parallel_state.is_pipeline_last_stage() and idx == len(data_parallel_group) - 1:
                        #print(f"当前rank是{torch.distributed.get_rank()},正在进行补全=====")
                        #print(f"长度是{len(self.data[:self.param_numel_count['word_embeddings']])}")
                        self.communication_handle = torch.distributed.all_reduce(
                            self.data[:self.param_numel_count["word_embeddings"]], group=group, async_op=self.overlap_grad_reduce
                        )
        elif parallel_state.is_pipeline_last_stage():
            for idx, group in data_parallel_group.items():
                if idx == len(data_parallel_group) - 1:
                    if parallel_state.is_independent_tp():
                        #print(f"当前rank是{torch.distributed.get_rank()},正在进行补全=====")
                        #print(f"start_idx={start_index}")
                        #print(f"data={len(self.data)}")
                        #print(f"word_embedding_start={len(self.data)-self.param_numel_count['final_word_embedding']}")
                        self.communication_handle = torch.distributed.all_reduce(
                            self.data[start_index:len(self.data)-self.param_numel_count["final_word_embedding"]], group=group, async_op=self.overlap_grad_reduce
                        )
                        self.communication_handle = torch.distributed.all_reduce(
                            self.data[len(self.data) - self.param_numel_count["final_word_embedding"]:],
                            group=group, async_op=self.overlap_grad_reduce
                        )
                    else:
                        self.communication_handle = torch.distributed.all_reduce(
                        self.data[start_index:], group=group, async_op=self.overlap_grad_reduce
                    )
                else:
                    end_index = start_index + self.param_numel_count["layer_weight"] * data_parallel_layer[idx]
                    self.communication_handle = torch.distributed.all_reduce(
                        self.data[start_index:end_index], group=group, async_op=self.overlap_grad_reduce
                    )
                    start_index = end_index
        else:
            for idx, group in data_parallel_group.items():
                end_index = start_index + self.param_numel_count["layer_weight"] * data_parallel_layer[idx]
                self.communication_handle = torch.distributed.all_reduce(
                    self.data[start_index:end_index], group=group, async_op=self.overlap_grad_reduce
                )
                start_index = end_index
        self.communication_issued = True

    def restore_grad(self):
        """
        把all_reduce后的梯度还原到原来的位置
        """
        rank = torch.distributed.get_rank()
        start_index = 0
        start_index_reduced = 0
        args = get_args()
        if rank == 4:
            start_index = self.param_numel_count["embedding_weight"] + self.param_numel_count["layer_weight"] * 4
        for i in range(2):
            # input_norm
            end_index = start_index + self.layer_weight_numel_count["input_norm_weight"] + \
                        self.layer_weight_numel_count[
                            "input_norm_bias"]
            end_index_reduced = start_index_reduced + self.layer_weight_numel_count["input_norm_weight"] + \
                                self.layer_weight_numel_count[
                                    "input_norm_bias"]
            self.data[start_index:end_index].copy_(self.top_half_data[start_index_reduced:end_index_reduced])
            # attention qkv weight,sharded by first dim
            start_index = end_index
            start_index_reduced = end_index_reduced
            end_index = start_index + self.layer_weight_numel_count["attention_qkv_weight"] // 2
            end_index_reduced = start_index_reduced + self.layer_weight_numel_count["attention_qkv_weight"] // 2
            self.data[start_index:end_index].copy_(self.top_half_data[start_index_reduced:end_index_reduced])
            start_index = end_index
            end_index = start_index + self.layer_weight_numel_count["attention_qkv_weight"] // 2
            self.data[start_index:end_index].copy_(self.bottom_half_data[start_index_reduced:end_index_reduced])
            # attention qkv bias,sharded by first dim
            start_index = end_index
            start_index_reduced = end_index_reduced
            end_index = start_index + self.layer_weight_numel_count["attention_qkv_bias"] // 2
            end_index_reduced = start_index_reduced + self.layer_weight_numel_count["attention_qkv_bias"] // 2
            self.data[start_index:end_index].copy_(self.top_half_data[start_index_reduced:end_index_reduced])
            start_index = end_index
            end_index = start_index + self.layer_weight_numel_count["attention_qkv_bias"] // 2
            self.data[start_index:end_index].copy_(self.bottom_half_data[start_index_reduced:end_index_reduced])
            # attention dense weight，shared by last dim
            start_index = end_index
            start_index_reduced = end_index_reduced
            end_index = start_index + self.layer_weight_numel_count["attention_dense_weight"]
            end_index_reduced = start_index_reduced + self.layer_weight_numel_count["attention_dense_weight"] // 2
            attention_dense_weight = self.data[start_index:end_index].view(args.hidden_size, args.hidden_size)
            attention_dense_weight[:, :attention_dense_weight.shape[1] // 2].copy_(
                self.top_half_data[start_index_reduced:end_index_reduced].view(args.hidden_size, args.hidden_size // 2))
            attention_dense_weight[:, attention_dense_weight.shape[1] // 2:].copy_(
                self.bottom_half_data[start_index_reduced:end_index_reduced].view(args.hidden_size,
                                                                                  args.hidden_size // 2))
            # attention dense bias，The bias within the tp group should be the same, and direct all reduce is sufficient
            start_index = end_index
            start_index_reduced = end_index_reduced
            end_index = start_index + self.layer_weight_numel_count["attention_dense_bias"]
            end_index_reduced = start_index_reduced + self.layer_weight_numel_count["attention_dense_bias"]
            self.data[start_index:end_index].copy_(self.top_half_data[start_index_reduced:end_index_reduced])
            # post attention norm weight and bias
            start_index = end_index
            start_index_reduced = end_index_reduced
            end_index = start_index + self.layer_weight_numel_count["post_attention_norm_weight"] + \
                        self.layer_weight_numel_count["post_attention_norm_bias"]
            end_index_reduced = start_index_reduced + self.layer_weight_numel_count["post_attention_norm_weight"] + \
                                self.layer_weight_numel_count["post_attention_norm_bias"]
            self.data[start_index:end_index].co0py_(self.top_half_data[start_index_reduced:end_index_reduced])
            # mlp h to 4h weight,sharded by first dim
            start_index = end_index
            start_index_reduced = end_index_reduced
            end_index = start_index + self.layer_weight_numel_count["mlp_dense_h_to_4h_weight"] // 2
            end_index_reduced = start_index_reduced + self.layer_weight_numel_count["mlp_dense_h_to_4h_weight"] // 2
            self.data[start_index:end_index].copy_(self.top_half_data[start_index_reduced:end_index_reduced])
            start_index = end_index
            end_index = start_index + self.layer_weight_numel_count["mlp_dense_h_to_4h_weight"] // 2
            self.data[start_index:end_index].copy_(self.bottom_half_data[start_index_reduced:end_index_reduced])
            # mlp h to 4h bias,sharded by first dim
            start_index = end_index
            start_index_reduced = end_index_reduced
            end_index = start_index + self.layer_weight_numel_count["mlp_dense_h_to_4h_bias"] // 2
            end_index_reduced = start_index_reduced + self.layer_weight_numel_count["mlp_dense_h_to_4h_bias"] // 2
            self.data[start_index:end_index].copy_(self.top_half_data[start_index_reduced:end_index_reduced])
            start_index = end_index
            end_index = start_index + self.layer_weight_numel_count["mlp_dense_h_to_4h_bias"] // 2
            self.data[start_index:end_index].copy_(self.bottom_half_data[start_index_reduced:end_index_reduced])
            # mlp 4h to h weight ,sharded by last dim
            start_index = end_index
            start_index_reduced = end_index_reduced
            end_index = start_index + self.layer_weight_numel_count["mlp_dense_4h_to_h_weight"]
            end_index_reduced = start_index_reduced + self.layer_weight_numel_count["mlp_dense_4h_to_h_weight"] // 2
            mlp_4h_to_h_weight = self.data[start_index:end_index].view(args.hidden_size, 4 * args.hidden_size)
            mlp_4h_to_h_weight[:, :mlp_4h_to_h_weight.shape[1] // 2].copy_(
                self.top_half_data[start_index_reduced:end_index_reduced].view(args.hidden_size, 2 * args.hidden_size))
            mlp_4h_to_h_weight[:, mlp_4h_to_h_weight.shape[1] // 2:].copy_(
                self.bottom_half_data[start_index_reduced:end_index_reduced].view(args.hidden_size,
                                                                                  2 * args.hidden_size))
            # mlp 4h to h bias ,sharded by last dim
            start_index = end_index
            start_index_reduced = end_index_reduced
            end_index = start_index + self.layer_weight_numel_count["mlp_dense_4h_to_h_bias"]
            end_index_reduced = start_index_reduced + self.layer_weight_numel_count["mlp_dense_4h_to_h_bias"]
            self.data[start_index:end_index].copy_(self.top_half_data[start_index_reduced:end_index_reduced])
            start_index = end_index
            start_index_reduced = end_index_reduced

    def shard_grad(self):
        """
        把dp组中单卡的梯度拆成TP分组后的结果，
        针对的情形是：
        有的DP组中某层的梯度只在一张卡上（未TP），其他DP组中该层做了TP，方便1对多的梯度聚合
        """
        assert not parallel_state.is_tensor_parallel(), "TP group should not shard grad to sync grad"
        start_index = 0
        rank = torch.distributed.get_rank()
        if rank == 4:
            start_index = self.param_numel_count["embedding_weight"] + self.param_numel_count["layer_weight"] * 4
        args = get_args()
        # 每一个layer层的切分
        # input_norm
        for i in range(2):
            end_index = start_index + self.layer_weight_numel_count["input_norm_weight"] + \
                        self.layer_weight_numel_count[
                            "input_norm_bias"]
            input_norm = self.data[start_index:end_index].clone()
            # attention qkv weight,sharded by first dim
            start_index = end_index
            end_index = start_index + self.layer_weight_numel_count["attention_qkv_weight"]
            attention_qkv_weight = self.data[start_index:end_index].clone()
            attention_qkv_weight_top_half = attention_qkv_weight[:attention_qkv_weight.shape[0] // 2]
            attention_qkv_weight_bottom_half = attention_qkv_weight[attention_qkv_weight.shape[0] // 2:]
            # attention qkv bias,sharded by first dim
            start_index = end_index
            end_index = start_index + self.layer_weight_numel_count["attention_qkv_bias"]
            attention_qkv_bias = self.data[start_index:end_index].clone()
            attention_qkv_bias_top_half = attention_qkv_bias[:attention_qkv_bias.shape[0] // 2]
            attention_qkv_bias_bottom_half = attention_qkv_bias[attention_qkv_bias.shape[0] // 2:]
            # attention dense weight，shared by last dim
            start_index = end_index
            end_index = start_index + self.layer_weight_numel_count["attention_dense_weight"]
            attention_dense_weight = self.data[start_index:end_index].clone().view(args.hidden_size, args.hidden_size)
            attention_dense_weight_top_half = attention_dense_weight[:, :attention_dense_weight.shape[1] // 2].flatten()
            attention_dense_weight_bottom_half = attention_dense_weight[:,
                                                 attention_dense_weight.shape[1] // 2:].flatten()
            # attention dense bias，The bias within the tp group should be the same, and direct all reduce is sufficient
            start_index = end_index
            end_index = start_index + self.layer_weight_numel_count["attention_dense_bias"]
            attention_dense_bias = self.data[start_index:end_index].clone()
            # post attention norm weight and bias
            start_index = end_index
            end_index = start_index + self.layer_weight_numel_count["post_attention_norm_weight"] + \
                        self.layer_weight_numel_count["post_attention_norm_bias"]
            post_attention_norm_weight_and_bias = self.data[start_index:end_index].clone()
            # mlp h to 4h weight,sharded by first dim
            start_index = end_index
            end_index = start_index + self.layer_weight_numel_count["mlp_dense_h_to_4h_weight"]
            mlp_h_to_4h_weight = self.data[start_index:end_index].clone()
            mlp_h_to_4h_weight_top_half = mlp_h_to_4h_weight[:mlp_h_to_4h_weight.shape[0] // 2]
            mlp_h_to_4h_weight_bottom_half = mlp_h_to_4h_weight[mlp_h_to_4h_weight.shape[0] // 2:]
            # mlp h to 4h bias,sharded by first dim
            start_index = end_index
            end_index = start_index + self.layer_weight_numel_count["mlp_dense_h_to_4h_bias"]
            mlp_h_to_4h_bias = self.data[start_index:end_index].clone()
            mlp_h_to_4h_bias_top_half = mlp_h_to_4h_bias[:mlp_h_to_4h_bias.shape[0] // 2]
            mlp_h_to_4h_bias_bottom_half = mlp_h_to_4h_bias[mlp_h_to_4h_bias.shape[0] // 2:]
            # mlp 4h to h weight ,sharded by last dim
            start_index = end_index
            end_index = start_index + self.layer_weight_numel_count["mlp_dense_4h_to_h_weight"]
            mlp_4h_to_h_weight = self.data[start_index:end_index].clone().view(args.hidden_size, 4 * args.hidden_size)
            mlp_4h_to_h_weight_top_half = mlp_4h_to_h_weight[:, :mlp_4h_to_h_weight.shape[1] // 2].flatten()
            mlp_4h_to_h_weight_bottom_half = mlp_4h_to_h_weight[:, mlp_4h_to_h_weight.shape[1] // 2:].flatten()
            # mlp 4h to h bias ,sharded by last dim
            start_index = end_index
            end_index = start_index + self.layer_weight_numel_count["mlp_dense_4h_to_h_bias"]
            mlp_4h_to_h_bias = self.data[start_index:end_index].clone()

            self.top_half_data = torch.cat(
                [self.top_half_data, input_norm, attention_qkv_weight_top_half, attention_qkv_bias_top_half,
                 attention_dense_weight_top_half, attention_dense_bias, post_attention_norm_weight_and_bias,
                 mlp_h_to_4h_weight_top_half, mlp_h_to_4h_bias_top_half, mlp_4h_to_h_weight_top_half, mlp_4h_to_h_bias],
                dim=0)
            self.bottom_half_data = torch.cat(
                [self.bottom_half_data, input_norm, attention_qkv_weight_bottom_half, attention_qkv_bias_bottom_half,
                 attention_dense_weight_bottom_half, attention_dense_bias, post_attention_norm_weight_and_bias,
                 mlp_h_to_4h_weight_bottom_half, mlp_h_to_4h_bias_bottom_half, mlp_4h_to_h_weight_bottom_half,
                 mlp_4h_to_h_bias],
                dim=0)
            start_index = end_index
        print(
            f"当前rank为{torch.distributed.get_rank()}，当前top_half_data为{self.top_half_data},长度为:{self.top_half_data.shape}++++++++++++++++++++++++++++++++++")
        print(
            f"当前rank为{torch.distributed.get_rank()}，当前bottom_half_data为{self.bottom_half_data},长度为:{self.bottom_half_data.shape}++++++++++++++++++++++++++++++++++")

    def finish_grad_sync(self):
        """
        Finishes grad sync (all-reduce or reduce-scatter) communication operation
        for this bucket.

        When overlap_grad_reduce is set to True, waits for asynchronous communication
        call to complete. When overlap_grad_reduce is set to False, makes synchronous call.
        """
        # If overlap_grad_reduce is False, start (and finish) synchronous communication call here.
        if not self.overlap_grad_reduce:
            self.start_grad_sync()
            return
        assert self.communication_handle is not None and self.communication_issued, (
            f'Communication call has not been issued for this bucket '
            f'({len(self.params_with_grad)}/{len(self.params)} params have grad available)'
        )
        self.communication_handle.wait()

    def register_grad_ready(self, param: torch.nn.Parameter):
        """
        Registers grads for the passed-in param to be "ready" for grad sync.

        When the number of microbatches is greater than 1, we only want to register
        grads as ready when processing the last microbatch and overlap_grad_reduce is True.
        """
        assert param in self.params, 'Param is not in the bucket'
        assert param not in self.params_with_grad, 'Cannot set grad twice'
        assert (
            self.overlap_grad_reduce
        ), 'register_grad_ready() should be called only when overlapping grad reduce'
        self.params_with_grad.add(param)
        # If all params in bucket have grads available, issue communication call.
        if len(self.params_with_grad) == len(self.params):
            self.start_grad_sync()


class GradBuffer:
    """
    Groups gradients into a contiguous buffer, and then breaks the buffer into buckets with
    roughly `bucket_size` parameters each.

    Arguments:
        dtype: Type of underlying tensor.
        params: List of parameters whose gradients are collated in the underlying tensor.
        data_parallel_group: Data-parallel process group.
        bucket_size: The rough size of each bucket in terms of number of parameters.
        param_to_name: Mapping from `torch.nn.Parameter` to name (for logging purposes).
        overlap_grad_reduce: If true, overlap communication with backprop computation by
            breaking up grads into buckets. If false, single synchronous communication call
            is used instead.
        use_distributed_optimizer: If true, issue reduce-scatter communication calls as part
            of distributed optimizer. If false, issue all-reduce communication calls.
    """

    def __init__(
            self,
            dtype: torch.dtype,
            params: List[torch.nn.Parameter],
            data_parallel_group: torch.distributed.ProcessGroup,
            bucket_size: int,
            param_to_name: Dict[torch.nn.Parameter, str],
            overlap_grad_reduce: bool,
            use_distributed_optimizer: bool,
            param_numel_count,
            layer_weight_numel_count,
    ):

        # Check that params are unique.
        unique_params = set()
        for param in params:
            assert param not in unique_params
            unique_params.add(param)
        del unique_params

        self.param_numel_count = param_numel_count
        self.layer_weight_numel_count = layer_weight_numel_count
        # Store attributes that will be needed later.
        self.dtype = dtype
        self.data_parallel_group = data_parallel_group
        # self.data_parallel_world_size = torch.distributed.get_world_size(
        #     group=self.data_parallel_group
        # )
        self.data_parallel_world_size = parallel_state.get_data_parallel_world_size()
        self.overlap_grad_reduce = overlap_grad_reduce
        self.use_distributed_optimizer = use_distributed_optimizer
        self.is_last_microbatch = True

        # Data structures to store underlying buckets and relevant indexing data.
        self.buckets = []
        self.param_to_bucket = {}  # Param -> bucket mapping.
        self.param_index_map = {}  # Param -> location in buffer mapping (used in dist. optimizer).

        def _pad_if_needed(data_index: int):
            """Pads data indices if using distributed optimizer (to ensure uniform sharding)."""
            if use_distributed_optimizer:
                return (
                        int(math.ceil(data_index / self.data_parallel_world_size))
                        * self.data_parallel_world_size
                )
            return data_index

        # First, figure out how many elements should be in the underlying buffer storage.
        # Note that if we need to split the buffer into smaller buckets, each of these
        # might need to be padded as well (if using the distributed optimizer).
        data_start_index = 0
        bucket_data_start_index = data_start_index
        bucket_params = set()
        self.bucket_indices = []
        bucket_id = 0
        for param in params[::-1]:
            # Iterate through parameters in reverse order to roughly follow backprop order,
            # and skip parameters that don't require gradients.
            if not param.requires_grad:
                continue
            this_numel = param.data.nelement()
            data_end_index = data_start_index + this_numel
            self.param_index_map[param] = (
                data_start_index,
                data_end_index,
                bucket_id,
            )
            bucket_params.add(param)

            # If we have enough elements already, form a new bucket.
            # If bucket_size is None, accumulate everything into a single bucket.

            # TODO: Remove len(bucket_params) > 1 when the final head that transforms token
            # representations from hidden space to vocabulary space is in a PyTorch module
            # whose forward method is called. If it is not and a bucket contains only this
            # one parameter, we get incorrect behavior (i.e., higher losses) since we do not
            # call the wait function on the bucket's all_gather_handle (we use forward pre-
            # hooks on PyTorch modules to do this when --overlap-param-gather is used).
            # As a temporary workaround, we make sure that no bucket has only one parameter.
            if bucket_size is not None:
                if (data_end_index - bucket_data_start_index) >= bucket_size and len(
                        bucket_params
                ) > 1:
                    data_end_index = _pad_if_needed(data_end_index)
                    self.bucket_indices.append((bucket_data_start_index, data_end_index))
                    bucket_data_start_index = data_end_index
                    bucket_params = set()
                    bucket_id += 1
            data_start_index = data_end_index

        # Add remaining params to a new bucket.
        if len(bucket_params) > 0:
            data_end_index = _pad_if_needed(data_end_index)
            self.bucket_indices.append((bucket_data_start_index, data_end_index))

        # Next, create underlying storage for buffer (with numel elements that includes
        # padding as necessary).
        self.numel = data_end_index
        print(f"当前rank为{torch.distributed.get_rank()}，当前numel为{self.numel}++++++++++++++++++++++++++++++++++")
        if use_distributed_optimizer:
            assert self.numel % self.data_parallel_world_size == 0
        self.data = torch.zeros(
            self.numel, dtype=self.dtype, device=torch.cuda.current_device(), requires_grad=False,
        )

        # Finally, map main_grad fields for each parameter with a .grad field.
        bucket_params = set()
        bucket_data_start_index = 0
        cur_bucket_id = 0
        for param in params[::-1]:
            if not param.requires_grad:
                continue
            data_start_index, data_end_index, bucket_id = self.param_index_map[param]
            param.main_grad = self._get(param.data.shape, data_start_index)
            if bucket_id != cur_bucket_id:
                bucket_data_end_index = _pad_if_needed(data_start_index)
                self._set_bucket(
                    bucket_params, bucket_data_start_index, bucket_data_end_index, cur_bucket_id
                )
                bucket_data_start_index = bucket_data_end_index
                bucket_params = set()
                assert cur_bucket_id + 1 == len(self.buckets)
                assert bucket_id == cur_bucket_id + 1
                cur_bucket_id = bucket_id
            bucket_params.add(param)

        # Add remaining params to a new bucket.
        if len(bucket_params) > 0:
            bucket_data_end_index = _pad_if_needed(data_end_index)
            self._set_bucket(
                bucket_params, bucket_data_start_index, bucket_data_end_index, cur_bucket_id
            )

        if not overlap_grad_reduce:
            assert len(bucket_params) == len(
                params
            ), 'All params should be in one bucket when overlap_grad_reduce is False'

        # Log buckets for all PP stages.
        if (
                parallel_state.get_data_parallel_rank(with_context_parallel=True) == 0
                and parallel_state.get_tensor_model_parallel_rank() == 0
        ):
            logger.info(
                f'Number of buckets for gradient all-reduce / reduce-scatter: {len(self.buckets)}'
            )
            for index, bucket in enumerate(self.buckets):
                numel = 0
                for param in bucket.params:
                    numel += param.data.nelement()
                logger.info(f'Params for bucket {index + 1} ({numel} elements):')
                for param in bucket.params:
                    logger.info(f'    {param_to_name[param]}')

    def _get(self, shape: torch.Size, start_index: int) -> torch.Tensor:
        """
        Return a tensor with the input `shape` as a view into the 1-D data starting at
        `start_index`.
        """
        end_index = start_index + shape.numel()
        assert end_index <= self.numel, 'Requested tensor is out of buffer range'
        buffer_tensor = self.data[start_index:end_index]
        buffer_tensor = buffer_tensor.view(shape)
        return buffer_tensor

    def _set_bucket(
            self,
            bucket_params: List[torch.nn.Parameter],
            start_index: int,
            end_index: int,
            bucket_id: int,
    ):
        """
        Helper function to create new bucket, add it to list of buckets, and
        also update param->bucket mapping.
        """

        # Assert that indices are correctly padded (if needed), and that bucket
        # position is same as originally computed.
        if self.use_distributed_optimizer:
            assert start_index % self.data_parallel_world_size == 0
            assert end_index % self.data_parallel_world_size == 0
        assert (start_index, end_index) == self.bucket_indices[bucket_id]

        # Get appropriate view into global GradBuffer.
        bucket_data = self._get(torch.Size([end_index - start_index]), start_index)
        bucket = Bucket(
            params=bucket_params,
            data=bucket_data,
            offset=start_index,
            data_parallel_group=self.data_parallel_group,
            data_parallel_world_size=self.data_parallel_world_size,
            overlap_grad_reduce=self.overlap_grad_reduce,
            use_distributed_optimizer=self.use_distributed_optimizer,
            param_numel_count=self.param_numel_count,
            layer_weight_numel_count=self.layer_weight_numel_count,
        )
        self.buckets.append(bucket)
        for bucket_param in bucket_params:
            assert bucket_param not in self.param_to_bucket
            self.param_to_bucket[bucket_param] = bucket

    def reset(self, zero_buffer):
        """
        Zero out the underlying buffer and reset all buckets in preparation for the next
        iteration of training.

        When zero_buffer is set to True, the underlying buffer is zeroed out.
        """
        if zero_buffer:
            self.data.zero_()
        for bucket in self.buckets:
            bucket.reset()
        self.is_last_microbatch = True

    def start_grad_sync(self):
        """
        Initiates grad sync (all-reduce or reduce-scatter) communication operations
        for all buckets in the grad buffer.

        When overlap_grad_reduce is set to True, dispatches asynchronous communication
        calls. When overlap_grad_reduce is set to False, calls synchronous
        communication ops.
        """
        for bucket in self.buckets:
            bucket.start_grad_sync()

    def finish_grad_sync(self):
        """
        Finishes grad sync (all-reduce or reduce-scatter) communication operations
        for all buckets in the grad buffer.

        When overlap_grad_reduce is set to True, waits for asynchronous communication
        calls to complete. When overlap_grad_reduce is set to False, calls synchronous
        communication ops.
        """
        for bucket in self.buckets:
            bucket.finish_grad_sync()

    def register_grad_ready(self, param: torch.nn.Parameter):
        """
        Registers grads for the passed-in param to be "ready" for grad sync.

        When the number of microbatches is greater than 1, we only want to register
        grads as ready when processing the last microbatch and overlap_grad_reduce is True.
        """
        assert (
            self.overlap_grad_reduce
        ), 'register_grad_ready() should only be called when overlap_grad_reduce is True'
        if self.is_last_microbatch:
            bucket = self.param_to_bucket[param]
            bucket.register_grad_ready(param)
