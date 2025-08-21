#ifndef CU_MATCH_KERNEL_EXECUTOR_CUH
#define CU_MATCH_KERNEL_EXECUTOR_CUH
#include <cu_match/device_defines.cuh>
#include <cu_match/subgraph_match_defines.h>

namespace cu_match {

class kernel_executor {

public:
    struct config_t {
        query_pattern_type q_type;
        join_algorithm_type j_type;
    };

    using kernel_fnsig = void(kernel_args);
    using kernel_fnptr = std::add_pointer_t<kernel_fnsig>;

    kernel_executor();
    ~kernel_executor() noexcept;
    void init(config_t const& cfg);
    GSTREAM_HOST_ONLY void LFTJ_kernel_invoker(kernel_args const& kargs, gstream::cuda::stream_type const& stream);

private:
    config_t _cfg;
};


} // namespace cu_match

#endif // CU_MATCH_KERNEL_EXECUTOR_CUH
