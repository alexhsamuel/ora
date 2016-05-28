#pragma once

#include <cassert>
#include <limits>
#include <iostream>

namespace aslib {

//------------------------------------------------------------------------------

using int128_t = __int128;
using uint128_t = unsigned __int128;

static_assert(sizeof(int128_t) == 16, "int128_t isn't 128 bits");
static_assert(sizeof(uint128_t) == 16, "uint128_t isn't 128 bits");

/*
 * Constructs an unisgned 128-bit integer out of high and low 64 bit parts.
 */
inline constexpr uint128_t
make_uint128(
  uint64_t hi,
  uint64_t lo)
{
  return ((uint128_t) hi) << 64 | lo;
}


//------------------------------------------------------------------------------


inline unsigned long
pow10(
  unsigned exp)
{
  // FIXME: Bogosity.
  static unsigned long pows[] = {
    (unsigned long) 1ul,
    (unsigned long) 10ul,
    (unsigned long) 100ul,
    (unsigned long) 1000ul,
    (unsigned long) 10000ul,
    (unsigned long) 100000ul,
    (unsigned long) 1000000ul,
    (unsigned long) 10000000ul,
    (unsigned long) 100000000ul,
    (unsigned long) 1000000000ul,
    (unsigned long) 10000000000ul, 
    (unsigned long) 100000000000ul, 
    (unsigned long) 1000000000000ul,
    (unsigned long) 10000000000000ul,
    (unsigned long) 100000000000000ul,
    (unsigned long) 1000000000000000ul,
    (unsigned long) 10000000000000000ul,
    (unsigned long) 100000000000000000ul,
    (unsigned long) 1000000000000000000ul,
    (unsigned long) 10000000000000000000ul,
  };
  assert(exp < sizeof(pows) / sizeof(unsigned long));
  return pows[exp];
}


/**
 * Returns true if 'val' is in the (closed) range ['min', 'max'].
 */
template<typename T>
inline constexpr bool
in_range(
  T min,
  T val,
  T max)
{
  return min <= val && val <= max;
}


/**
 * Returns true if 'val' is in the half-open range ['min', 'bound').
 */
template<typename T>
inline constexpr bool
in_interval(
  T min,
  T val,
  T bound)
{
  return min <= val && val < bound;
}


// FIXME: Not used. (?)  And probably doesn't work for __int128.
/*
 * True if `val` overflows when conveted from integer types `FROM` to `TO`.
 */
template<typename TO, typename FROM>
inline bool constexpr
overflows(
  FROM val) 
{
  static_assert(
    std::numeric_limits<FROM>::is_integer, 
    "overflows() for integer types only");
  static_assert(
    std::numeric_limits<TO>::is_integer, 
    "overflows() for integer types only");

  return
    std::numeric_limits<TO>::is_signed
    ? 
       (   !std::numeric_limits<FROM>::is_signed 
        && (uintmax_t) val > (uintmax_t) INTMAX_MAX) 
    || (intmax_t) val < (intmax_t) std::numeric_limits<TO>::min() 
    || (intmax_t) val > (intmax_t) std::numeric_limits<TO>::max()
    : 
       val < 0
    || (uintmax_t) val > (uintmax_t) std::numeric_limits<TO>::max();
}


template<typename T>
inline T
round_div(
  T num,
  T den)
{
  return (num + den / 2) / den;
}


template<class T0, class T1>
inline T1
rescale_int(
  T0 const val,
  T0 const old_den,
  T1 const new_den)
{
  // FIXME: Careful!
  if (old_den % new_den == 0)
    return round_div(val, old_den / (T0) new_den);
  else if (new_den % old_den == 0)
    return val * (new_den / old_den);
  else if (sizeof(T0) >= sizeof(T1))
    return round_div((T0) val * (T0) new_den, (T0) old_den);
  else
    return round_div((T1) val * (T1) new_den, (T1) old_den);
}


/**
 * Rescales a value from one denominator to another, rounding if necessary.
 */
// FIXME: Is this really necessary?  Won't the above be inlined well enough?
template<typename T, T OLD_DEN, T NEW_DEN>
inline T
rescale_int(
  T val)
{
  if (OLD_DEN % NEW_DEN == 0)
    return round_div(val, OLD_DEN / NEW_DEN);
  else if (NEW_DEN % OLD_DEN == 0)
    return val * (NEW_DEN / OLD_DEN);
  else
    return round_div((int128_t) (val * NEW_DEN), (int128_t) OLD_DEN);
}


//------------------------------------------------------------------------------

inline bool add_overflow(unsigned int       a, unsigned int       b, unsigned int      & r) { return __builtin_uadd_overflow  (a, b, &r); }
inline bool add_overflow(unsigned long      a, unsigned long      b, unsigned long     & r) { return __builtin_uaddl_overflow (a, b, &r); }
inline bool add_overflow(unsigned long long a, unsigned long long b, unsigned long long& r) { return __builtin_uaddll_overflow(a, b, &r); }

inline bool add_overflow(         int       a,          int       b,          int      & r) { return __builtin_sadd_overflow  (a, b, &r); }
inline bool add_overflow(         long      a,          long      b,          long     & r) { return __builtin_saddl_overflow (a, b, &r); }
inline bool add_overflow(         long long a,          long long b,          long long& r) { return __builtin_saddll_overflow(a, b, &r); }

inline bool sub_overflow(unsigned short     a, unsigned short     b, unsigned short    & r) { return a < b || ((r = a - b) && false); }
inline bool sub_overflow(unsigned int       a, unsigned int       b, unsigned int      & r) { return __builtin_usub_overflow  (a, b, &r); }
inline bool sub_overflow(unsigned long      a, unsigned long      b, unsigned long     & r) { return __builtin_usubl_overflow (a, b, &r); }
inline bool sub_overflow(unsigned long long a, unsigned long long b, unsigned long long& r) { return __builtin_usubll_overflow(a, b, &r); }

inline bool sub_overflow(         int       a,          int       b,          int      & r) { return __builtin_ssub_overflow  (a, b, &r); }
inline bool sub_overflow(         long      a,          long      b,          long     & r) { return __builtin_ssubl_overflow (a, b, &r); }
inline bool sub_overflow(         long long a,          long long b,          long long& r) { return __builtin_ssubll_overflow(a, b, &r); }

inline bool mul_overflow(unsigned int       a, unsigned int       b, unsigned int      & r) { return __builtin_umul_overflow  (a, b, &r); }
inline bool mul_overflow(unsigned long      a, unsigned long      b, unsigned long     & r) { return __builtin_umull_overflow (a, b, &r); }
inline bool mul_overflow(unsigned long long a, unsigned long long b, unsigned long long& r) { return __builtin_umulll_overflow(a, b, &r); }

inline bool mul_overflow(         int       a,          int       b,          int      & r) { return __builtin_smul_overflow  (a, b, &r); }
inline bool mul_overflow(         long      a,          long      b,          long     & r) { return __builtin_smull_overflow (a, b, &r); }
inline bool mul_overflow(         long long a,          long long b,          long long& r) { return __builtin_smulll_overflow(a, b, &r); }

// FIXME
inline bool
add_overflow(
  uint128_t a,
  uint128_t b,
  uint128_t& r)
{
  r = a + b;
  return false;
}

// FIXME
inline bool
mul_overflow(
  uint128_t a,
  uint128_t b,
  uint128_t& r)
{
  r = a * b;
  return false;
}


//------------------------------------------------------------------------------

}  // namespace aslib

//------------------------------------------------------------------------------

namespace std {

// FIXME: Hack to print uint128_t.
inline std::ostream&
operator<<(
  std::ostream& os,
  aslib::uint128_t x)
{
  char buf[40];
  char* p = &buf[39];
  *p = 0;
  if (x == 0)
    *--p = '0';
  else
    while (x > 0) {
      *--p = '0' + x % 10;
      x /= 10;
    }
  os << p;
  return os;
}


}
