#pragma once

#include "ora/lib/exc.hh"
#include "ora/lib/math.hh"
#include "ora/time_math.hh"
#include "ora/time_nex.hh"
#include "ora/time_type.hh"

namespace ora {
namespace time {

//------------------------------------------------------------------------------
// Factory functions
//------------------------------------------------------------------------------

template<class TIME=Time>
inline TIME 
from_offset(
  typename TIME::Offset const offset)
{ 
  return TIME::from_offset(offset);
}


template<class TIME=Time>
inline TIME
from_timespec(
  timespec const ts)
{
  return from_offset(timespec_to_offset<TIME>(ts));
}


//------------------------------------------------------------------------------

/*
 * Returns the closest UNIX epoch time.
 *
 * Returns the rounded (signed) number of seconds since 1970-01-01T00:00:00Z.
 */
template<class TIME>
inline int64_t
get_epoch_time(
  TIME const time)
{
  return convert_offset(
    time.get_offset(), TIME::DENOMINATOR, TIME::BASE,
    1, DATENUM_UNIX_EPOCH);
}


template<class TIME=Time>
inline TIME
now()
{
  auto const ts = now_timespec();
  return 
      ts.tv_nsec >= 0
    ? TIME::from_offset(timespec_to_offset<TIME>(ts)) 
    : TIME::INVALID;
}


//------------------------------------------------------------------------------
// Comparisons
//------------------------------------------------------------------------------

template<class TIME>
inline bool
equal(
  TIME const time0,
  TIME const time1)
{
  ensure_valid(time0);
  ensure_valid(time1);
  return time0.get_offset() == time1.get_offset();
}


template<class TIME>
inline bool
before(
  TIME const time0,
  TIME const time1)
{
  ensure_valid(time0);
  ensure_valid(time1);
  return time0.get_offset() < time1.get_offset();
}


template<class TIME>
inline int
compare(
  TIME const time0,
  TIME const time1)
{
  ensure_valid(time0);
  ensure_valid(time1);
  return lib::compare(time0.get_offset(), time1.get_offset());
}


//------------------------------------------------------------------------------
// Arithemtic with seconds
//------------------------------------------------------------------------------

/*
 * Shifts `time` forward by `seconds`.
 */
template<class TIME>
inline TIME
seconds_after(
  TIME const time,
  double const seconds)
{
  using Offset = typename TIME::Offset;
  ensure_valid(time);
  // FIXME: Check for overflow.
  return from_offset<TIME>(
    time.get_offset() + (Offset) round(seconds * TIME::DENOMINATOR));
}


/*
 * Shifts `time` backward by `seconds`.
 */
template<class TIME>
inline TIME
seconds_before(
  TIME const time,
  double const seconds)
{
  using Offset = typename TIME::Offset;
  ensure_valid(time);
  // FIXME: Check for overflow.
  return from_offset<TIME>(
    time.get_offset() - (Offset) round(seconds * TIME::DENOMINATOR));
}


/*
 * The number of seconds between `time0` and `time1.
 */
template<class TIME>
inline double
seconds_between(
  TIME const time0,
  TIME const time1)
{
  auto const off0 = time0.get_offset();
  auto const off1 = time1.get_offset();
  // Needs to work for unsigned offsets.
  return
    off1 >= off0
    ?   (off1 - off0) * TIME::RESOLUTION
    : -((off0 - off1) * TIME::RESOLUTION);
}


//------------------------------------------------------------------------------
// Addition and subtraction
//------------------------------------------------------------------------------

template<class TRAITS> inline TimeType<TRAITS> operator+(TimeType<TRAITS> const t, double const secs)
  { return seconds_after(t, secs); }
template<class TRAITS> inline TimeType<TRAITS> operator-(TimeType<TRAITS> const t, double const secs)
  { return seconds_before(t, secs); }
template<class TRAITS> inline double operator-(TimeType<TRAITS> const t1, TimeType<TRAITS> const t0)
  { return seconds_between(t0, t1); } 

template<class TRAITS> inline TimeType<TRAITS> operator+=(TimeType<TRAITS>& t, int const secs) 
  { return t = t + secs; }
template<class TRAITS> inline TimeType<TRAITS> operator++(TimeType<TRAITS>& t) 
  { return t = t + 1; }
template<class TRAITS> inline TimeType<TRAITS> operator++(TimeType<TRAITS>& t, int /* tag */) 
  { auto old = t; t = t + 1; return old; }
template<class TRAITS> inline TimeType<TRAITS> operator-=(TimeType<TRAITS>& t, int const secs) 
  { return t = t - secs; }
template<class TRAITS> inline TimeType<TRAITS> operator--(TimeType<TRAITS>& t) 
  { return t = t - 1; }
template<class TRAITS> inline TimeType<TRAITS> operator--(TimeType<TRAITS>& t, int /* tag */) 
  { auto old = t; t = t - 1; return old; }

//------------------------------------------------------------------------------

}  // namespace time
}  // namespace ora

