#include <cu_match/kernel_executor.cuh>
#include <cu_match/device_defines.cuh>
#include <gstream/cuda_env.h>
#include <cooperative_groups.h>

namespace cu_match {

kernel_executor::kernel_executor() {}

kernel_executor::~kernel_executor() {}

void kernel_executor::init(config_t const& cfg) {
    _cfg = cfg;
}

GSTREAM_HOST_ONLY
void kernel_executor::LFTJ_kernel_invoker(
        kernel_args const& kargs, 
        gstream::cuda::stream_type const& stream) {
    
    if (_cfg.j_type == join_algorithm_type::ATTRIBUTE_JOIN) {
        if (_cfg.q_type == NORMAL_PATTERN) {
            GSTREAM_INVOKE_CUDA_KERNEL(
                LFTJ_kernel_basic,
                MaxBlocks,
                NumThreads,
                ShardMemSize,
                static_cast<cudaStream_t>(stream)
            ) (kargs);
        }
        else if (_cfg.q_type == (NORMAL_PATTERN | CONSTRAINT_PATTERN)) {
            GSTREAM_INVOKE_CUDA_KERNEL(
                LFTJ_kernel_constraint,
                MaxBlocks,
                NumThreads,
                ShardMemSize,
                static_cast<cudaStream_t>(stream)
            ) (kargs);
        }
        else if (_cfg.q_type == (NORMAL_PATTERN | OPTIONAL_PATTERN)) {
            GSTREAM_INVOKE_CUDA_KERNEL(
                LFTJ_kernel_optional,
                MaxBlocksOpt,
                NumThreadsOpt,
                ShardMemSize,
                static_cast<cudaStream_t>(stream)
            ) (kargs);
        }
        else if (_cfg.q_type == (NORMAL_PATTERN | NEGATIVE_PATTERN)) {
            GSTREAM_INVOKE_CUDA_KERNEL(
                LFTJ_kernel_negative,
                MaxBlocksOpt,
                NumThreadsOpt,
                ShardMemSize,
                static_cast<cudaStream_t>(stream)
            ) (kargs);
        }
        else if (_cfg.q_type == (NORMAL_PATTERN | CONSTRAINT_PATTERN | OPTIONAL_PATTERN)) { // unstable
            GSTREAM_INVOKE_CUDA_KERNEL(
                LFTJ_kernel_optional_constraint,
                MaxBlocksOpt,
                NumThreadsOpt,
                ShardMemSize,
                static_cast<cudaStream_t>(stream)
            ) (kargs);
        }
        else if (_cfg.q_type == (NORMAL_PATTERN | CONSTRAINT_PATTERN | NEGATIVE_PATTERN)) {
            GSTREAM_INVOKE_CUDA_KERNEL(
                LFTJ_kernel_negative_constraint,
                MaxBlocksOpt,
                NumThreadsOpt,
                ShardMemSize,
                static_cast<cudaStream_t>(stream)
            ) (kargs);
        }
        else {
            printf("Not supported yet\n");
            exit(1);
        }
    }
    else if (_cfg.j_type == join_algorithm_type::TABLE_JOIN) {
        if (_cfg.q_type == NORMAL_PATTERN) {
            GSTREAM_INVOKE_CUDA_KERNEL(
                MJ_kernel_basic,
                MaxBlocksOpt,
                NumThreadsOpt,
                ShardMemSize,
                static_cast<cudaStream_t>(stream)
            ) (kargs);
        }
        else {
            printf("Not supported yet\n");
            exit(1);
        }
    }
}

} // namespace cu_match


  
  