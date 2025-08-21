#ifndef GSTREAM_CUDA_API_H
#define GSTREAM_CUDA_API_H
#include <gstream/internal_defines.h>
#include <gstream/cuda_env.h>
#include <ash/config/osdetect.h>
#include <stdint.h>
#include <assert.h>
#include <memory>
#include <string>

namespace gstream {
namespace cuda {

uint32_t get_cuda_last_error();

bool  device_count(int* out);
bool  set_device(int device_id) noexcept;
void* device_malloc(size_t size);
bool  device_free(void* p) noexcept;
stream_type* create_streams(size_t count);
bool destory_streams(stream_type* streams, size_t count) noexcept;

constexpr int UnexpectedCudaReturnValue = -1;
constexpr int StreamStateIdle           = 1;
constexpr int StreamStateBusy           = 2;
int           is_idle_stream(stream_type const& stream);

bool stream_synchronize(stream_type const& stream);
bool device_synchronize() noexcept;

std::string get_device_name(device_id_t device_id);

bool h2dcpy(void* dst, void* src, size_t size);
bool h2dcpy_async(void* dst, void* src, size_t size, stream_type const& stream);
bool d2hcpy(void* dst, void* src, size_t size);
bool d2hcpy_async(void* dst, void* src, size_t size, stream_type const& stream);

bool device_memset(void* dst, int value, size_t count);
bool device_memset_async(void* dst, int value, size_t count, stream_type const& stream);

void* pinned_malloc(size_t size);
bool  pinned_free(void* p) noexcept;

#ifdef ASH_ENV_WINDOWS
#define GSTREAM_CUDART_CB __stdcall
#else
#define GSTREAM_CUDART_CB 
#endif // PLATFORM
using stream_callback_t = void(GSTREAM_CUDART_CB*)(void* /*usr*/);
bool add_host_function(stream_type const& stream, stream_callback_t fn, void* usr_data);

} // namespace cuda

struct cuda_malloc_deleter {
    void operator()(void* m) const {
        bool r = cuda::device_free(m);
        assert(r && "cuda::device_free failure!");
        _ash_unused(r);
    }
};

using cuda_raii_buffer = std::unique_ptr<void, cuda_malloc_deleter>;

inline cuda_raii_buffer make_cuda_raii_buffer(size_t const size) noexcept {
    return cuda_raii_buffer{ cuda::device_malloc(size) };
}

struct pinned_malloc_deleter {
    void operator()(void* m) const {
        bool r = cuda::pinned_free(m);
        assert(r && "cuda::pinned_free failure!");
        _ash_unused(r);
    }
};

using pinned_raii_buffer = std::unique_ptr<void, pinned_malloc_deleter>;

inline pinned_raii_buffer make_pinned_raii_buffer(size_t const size) noexcept {
    return pinned_raii_buffer{ cuda::pinned_malloc(size) };
}

} // namespace gstream

#endif // GSTREAM_CUDA_API_H
