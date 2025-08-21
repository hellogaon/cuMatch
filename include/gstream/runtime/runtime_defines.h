#ifndef GSTREAM_RUNTIME_TYPES_H
#define GSTREAM_RUNTIME_TYPES_H
#include <gstream/qrydef.h>

namespace gstream_runtime {

using GStream = gstream::drivers::gstream_instance;
using SuperstepReport = gstream::drivers::superstep_report;
using GStreamQuery = gstream::drivers::gstream_query;
using GridFileStream = gstream::drivers::grid_file_stream;

using GStreamReturnCode = gstream::gstream_return_code;
using GStreamConfig = gstream::drivers::gstream_config;
using GridTxDesc = gstream::drivers::grid_transaction_descriptor;
using WriteBufferDesc = gstream::write_buffer_descriptor;

using KernelLaunchParameters = gstream::kernel_launch_parameters;
using SleafPtrArr = gstream::readonly_sleaf_ptrarr;
using SleafPtr = gstream::sleaf_cptr;
using F24ShardPtrArr = gstream::readonly_f24_shard_ptrarr;
using F24Shard = gstream::flip24_shard;

using CudaUInt64 = gstream::cuda_uint64_t;
using DeviceStream = gstream::cuda::stream_type;
using LocalAddr = gstream::laddr_t;
using ColumnPointer = gstream::grid_format::colptr_t;

using gstream::drivers::NoConcurrentTxLimit;
using gstream::grid_format::GridBlockWidth;

using MemoryAllocationStrategy = gstream::memory_allocation_strategy;
using MemoryAllocationMode = gstream::memory_allocation_strategy::alloc_mode;

} // namespace gstream_runtime

#endif // !GSTREAM_RUNTIME_TYPES_H
