#ifndef GSTREAM_FLIP24_ARRAY_H
#define GSTREAM_FLIP24_ARRAY_H
#include <type_traits>
#include <stdint.h>
#include <gstream/cuda_env.h>
#include <assert.h>
#include <string.h> // memset

namespace gstream {

#pragma pack(push, 1)
struct _24bit_chunk {
    char data[3];
};
#pragma pack(pop)

template <typename T>
GSTREAM_DEVICE_COMPATIBLE uint64_t flip24_high_bit_offset(T index) {
    return static_cast<uint64_t>(((index >> 1) << 1) + ((index >> 1) << 2) + (index & 1) + 2);
}

template <typename T>
GSTREAM_DEVICE_COMPATIBLE uint64_t flip24_low_bit_offset(T index) {
    return static_cast<uint64_t>(((index >> 1) << 1) + ((index >> 1) << 2) + ((index & 1) << 2));
}

GSTREAM_DEVICE_COMPATIBLE ASH_FORCEINLINE
uint32_t read_flip24_element(void const* arr_, uint64_t const& index) {
    uint64_t const base_addr = ((index >> 1) << 1) + ((index >> 1) << 2);
    uint64_t const is_odd = index & 1;
    uint64_t const addr_hi = base_addr + is_odd + 2;
    uint64_t const addr_lo = (base_addr + (is_odd << 2));
    uint8_t const* arr = static_cast<uint8_t const*>(arr_);
    uint32_t r = 0;
    r |= static_cast<uint8_t>(*(arr + addr_hi));
    r <<= 16;
    r |= *reinterpret_cast<uint16_t const*>(arr + addr_lo);
    return r;
}

GSTREAM_DEVICE_COMPATIBLE ASH_FORCEINLINE
uint32_t read_flip24_element2(void const* arr_, uint32_t const& idx) {
    //uint32_t const addr = (idx << 1) + idx; // idx * 3
    uint32_t const is_odd = idx & 1;
    uint8_t const* arr = static_cast<uint8_t const*>(arr_);
    uint8_t const H = *(arr + ((is_odd ^ 1) << 1));
    uint16_t const L = *reinterpret_cast<uint16_t const*>(arr + is_odd);
    uint32_t R = H;
    R <<= 16;
    R |= L;
    return R;
}

GSTREAM_DEVICE_COMPATIBLE ASH_FORCEINLINE
uint8_t read_flip24_high_bits(void const* arr_, uint64_t const& index) {
    uint64_t const base_addr = ((index >> 1) << 1) + ((index >> 1) << 2);
    uint64_t const is_odd = index & 1;
    uint64_t const addr_hi = base_addr + is_odd + 2;
    uint8_t const* arr = static_cast<uint8_t const*>(arr_);
    uint8_t r = 0;
    r |= static_cast<uint8_t>(*(arr + addr_hi));
    return r;
}

GSTREAM_DEVICE_COMPATIBLE ASH_FORCEINLINE
uint16_t read_flip24_low_bits(void const* arr_, uint64_t const& index) {
    uint64_t const base_addr = ((index >> 1) << 1) + ((index >> 1) << 2);
    uint64_t const is_odd = index & 1;
    uint64_t const addr_lo = (base_addr + (is_odd << 2));
    uint8_t const* arr = static_cast<uint8_t const*>(arr_);
    uint16_t r = 0;
    r |= *reinterpret_cast<uint16_t const*>(arr + addr_lo);
    return r;
}

GSTREAM_DEVICE_COMPATIBLE ASH_FORCEINLINE
void write_flip24_element
(void* arr_, uint64_t const& index, uint16_t const lo, uint8_t const hi) {
    uint64_t const base_addr = ((index >> 1) << 1) + ((index >> 1) << 2);
    uint64_t const is_odd = index & 1;
    uint64_t const addr_hi = base_addr + is_odd + 2;
    uint64_t const addr_lo = (base_addr + (is_odd << 2));
    uint8_t* arr = static_cast<uint8_t*>(arr_);
    *reinterpret_cast<uint16_t*>(arr + addr_lo) = lo;
    *(arr + addr_hi) = hi;
}

template <typename I, typename Addr = uint64_t>
std::enable_if_t<std::is_integral<I>::value>
swap_flip24_elements_xor(void* arr_, I const a, I const b) {
    assert(a != b);
    uint8_t* arr = static_cast<uint8_t*>(arr_);
    Addr const a_base = ((a >> 1) << 1) + ((a >> 1) << 2);
    uint8_t const is_a_odd = static_cast<uint8_t>(a & 1);
    uint16_t& a_lo = *static_cast<uint16_t*>(arr + a_base + is_a_odd + 2);
    uint8_t& a_hi = *static_cast<uint8_t*>(arr + a_base + (is_a_odd << 2));

    Addr const b_base = ((b >> 1) << 1) + ((b >> 1) << 2);
    uint8_t const is_b_odd = static_cast<uint8_t>(b & 1);
    uint16_t& b_lo = *static_cast<uint16_t*>(arr + b_base + is_b_odd + 2);
    uint8_t& b_hi = *static_cast<uint8_t*>(arr + b_base + (is_b_odd << 2));

    a_lo ^= b_lo;
    b_lo ^= a_lo;
    a_lo ^= b_lo;

    a_hi ^= b_hi;
    b_hi ^= a_hi;
    a_hi ^= b_hi;
}

template <typename I, typename Addr = uint64_t>
std::enable_if_t<std::is_integral<I>::value>
swap_flip24_elements(void* arr_, I const a, I const b) {
    if (MIXX_UNLIKELY(a == b))
        return;
    auto arr = static_cast<uint8_t*>(arr_);
    Addr const a_base = ((a >> 1) << 1) + ((a >> 1) << 2);
    auto const is_a_odd = static_cast<uint8_t>(a & 1);
    auto& a_lo = *reinterpret_cast<uint16_t*>(arr + a_base + is_a_odd + 2);
    auto& a_hi = *static_cast<uint8_t*>(arr + a_base + (is_a_odd << 2));

    Addr const b_base = ((b >> 1) << 1) + ((b >> 1) << 2);
    auto const is_b_odd = static_cast<uint8_t>(b & 1);
    auto& b_lo = *reinterpret_cast<uint16_t*>(arr + b_base + is_b_odd + 2);
    auto& b_hi = *static_cast<uint8_t*>(arr + b_base + (is_b_odd << 2));

    uint16_t const t_lo = a_lo;
    uint8_t  const t_hi = a_hi;
    a_lo = b_lo;
    a_hi = b_hi;
    b_lo = t_lo;
    b_hi = t_hi;
}

inline void flip24_memset(void* dst, uint16_t const lo, uint8_t hi, uint64_t size) {
    for (uint64_t i = 0; i < size; ++i) {
        write_flip24_element(dst, i, lo, hi);
    }
}

inline void flip24_zerofill(void* dst, uint64_t length) {
    memset(dst, 0, 3 * length);
}

#pragma pack(push, 1)
struct flip24_element {
    char data[3];
};
#pragma pack(pop)

} // !namespace gstream
#endif // GSTREAM_FLIP24_VECTOR_H
