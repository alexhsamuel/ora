#pragma once

#include <cassert>

namespace alxs {
namespace cron {

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


template<typename T>
inline T
round_div(
  T num,
  T den)
{
  return (num + den / 2) / den;
}


template<typename T>
inline T
rescale_int(
  T val,
  T old_den,
  T new_den)
{
  if (old_den % new_den == 0)
    return round_div(val, old_den / new_den);
  else if (new_den % old_den == 0)
    return val * (new_den / old_den);
  else
    return round_div((intmax_t) val * new_den, (intmax_t) old_den);
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
    return round_div((intmax_t) (val * NEW_DEN), (intmax_t) OLD_DEN);
}


//------------------------------------------------------------------------------

}  // namespace cron
}  // namespace alxs

