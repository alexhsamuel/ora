#pragma once

#include "math.hh"

namespace ora {
namespace lib {

//------------------------------------------------------------------------------
// Functions
//------------------------------------------------------------------------------

/* 
 * Advances pointer `p` by `step` bytes. 
 */
template<class T>
inline T*
step(
  T* const p,
  int const step)
{
  return (T*) (((char*) p) + step);
}


//------------------------------------------------------------------------------

template<unsigned char SIZE>
void
copy(
  void const* const src,
  void* const dst);


template<>
inline void
copy<1>(
  void const* const src,
  void* const dst)
{
  *(uint8_t*) dst = *(uint8_t const*) src;
}


template<>
inline void
copy<2>(
  void const* const src,
  void* const dst)
{
  *(uint16_t*) dst = *(uint16_t const*) src;
}


template<>
inline void
copy<4>(
  void const* const src,
  void* const dst)
{
  *(uint32_t*) dst = *(uint32_t const*) src;
}


template<>
inline void
copy<8>(
  void const* const src,
  void* const dst)
{
  *(uint64_t*) dst = *(uint64_t const*) src;
}


template<>
inline void
copy<16>(
  void const* const src,
  void* const dst)
{
  *(uint128_t*) dst = *(uint128_t const*) src;
}


template<unsigned char SIZE>
void
copy_swapped(
  void const* const src,
  void* const dst);


template<>
inline void
copy_swapped<1>(
  void const* const src,
  void* const dst)
{
  *(uint8_t*) dst = *(uint8_t const*) src;
}


template<>
inline void
copy_swapped<2>(
  void const* const src,
  void* const dst)
{
  *(uint16_t*) dst = __builtin_bswap16(*(uint16_t const*) src);
}


template<>
inline void
copy_swapped<4>(
  void const* const src,
  void* const dst)
{
  *(uint32_t*) dst = __builtin_bswap32(*(uint32_t const*) src);
}


template<>
inline void
copy_swapped<8>(
  void const* const src,
  void* const dst)
{
  *(uint64_t*) dst = __builtin_bswap64(*(uint64_t const*) src);
}


template<>
inline void
copy_swapped<16>(
  void const* const src,
  void* const dst)
{
  uint64_t const* const s = (uint64_t*) src;
  uint64_t* const d = (uint64_t*) dst;
  *(d + 1) = __builtin_bswap64(*s);
  *d = __builtin_bswap64(*(s + 1));
}


//------------------------------------------------------------------------------

}  // namespace lib
}  // namespace ora

