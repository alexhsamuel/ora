#pragma once

#include "aslib/exc.hh"
#include "aslib/math.hh"
#include "aslib/printable.hh"
#include "cron/date.hh"
#include "cron/daytime.hh"
#include "cron/time_math.hh"
#include "cron/time_type.hh"

namespace cron {
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
  return Unix64Time(time).get_offset();
}


template<class TIME=Time>
inline TIME
now()
{
  auto const ts = now_timespec();
  return 
      ts.tv_nsec >= 0
    ? from_offset<TIME>(timespec_to_offset<TIME>(ts)) 
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
  return aslib::compare(time0.get_offset(), time1.get_offset());
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
  ensure_valid(time0);
  ensure_valid(time1);
  return ((double) time1.get_offset() - time0.get_offset()) / TIME::DENOMINATOR;
}


//------------------------------------------------------------------------------
// Non-throwing versions
//------------------------------------------------------------------------------

namespace nex {

template<class TIME=Time>
inline TIME
from_offset(
  typename TIME::Offset const offset)
  noexcept
{
  return 
      in_range(TIME::Traits::min, offset, TIME::Traits::max)
    ? time::from_offset(offset)
    : TIME::INVALID;
}


template<class TIME=Time>
inline TIME
from_timespec(
  timespec const ts)
{
  return nex::from_offset(timespec_to_offset<TIME>(ts));
}


/*
 * Returns the closest UNIX epoch time.
 *
 * Returns the rounded (signed) number of seconds since 1970-01-01T00:00:00Z.
 */
template<class TIME>
inline EpochTime
get_epoch_time(
  TIME const time)
  noexcept
{
  return
      time.is_valid() 
    ? Unix64Time(time).get_offset()
    : EPOCH_TIME_INVALID;
}


template<class TIME>
inline bool
equal(
  TIME const time0,
  TIME const time1)
  noexcept
{
  return time0.offset_ == time1.offset_;
}


template<class TIME>
inline bool
before(
  TIME const time0,
  TIME const time1)
  noexcept
{
  if (nex::equal(time0, time1))
    return false;
  else if (time0.is_invalid())
    return true;
  else if (time1.is_invalid())
    return false;
  else if (time0.is_missing())
    return true;
  else if (time1.is_missing())
    return false;
  else
    return time0.get_offset() < time1.get_offset();
}


template<class TIME>
inline int
compare(
  TIME const time0,
  TIME const time1)
{
  return
      nex::equal(time0, time1) ? 0
    : nex::before(time0, time1) ? -1
    : 1;
}


}  // namespace nex

//------------------------------------------------------------------------------
// Comparison operators
//------------------------------------------------------------------------------

template<class T0, class T1> inline bool operator==(TimeType<T0> const t0, TimeType<T1> const t1) noexcept
  { return nex::equal(t0, TimeType<T0>(t1)); }
template<class T0, class T1> inline bool operator!=(TimeType<T0> const t0, TimeType<T1> const t1) noexcept
  { return !nex::equal(t0, TimeType<T0>(t1)); }
template<class T0, class T1> inline bool operator< (TimeType<T0> const t0, TimeType<T1> const t1) noexcept
  { return nex::before(t0, TimeType<T0>(t1)); }
template<class T0, class T1> inline bool operator> (TimeType<T0> const t0, TimeType<T1> const t1) noexcept
  { return nex::before(TimeType<T0>(t1), t0); }
template<class T0, class T1> inline bool operator<=(TimeType<T0> const t0, TimeType<T1> const t1) noexcept
  { return !nex::before(TimeType<T0>(t1), t0); }
template<class T0, class T1> inline bool operator>=(TimeType<T0> const t0, TimeType<T1> const t1) noexcept
  { return !nex::before(t0, TimeType<T0>(t1)); }

//------------------------------------------------------------------------------
// Addition and subtraction
//------------------------------------------------------------------------------

template<class TRAITS> inline TimeType<TRAITS> operator+(TimeType<TRAITS> const t, double const secs)
  { return seconds_after(t, secs); }
template<class TRAITS> inline TimeType<TRAITS> operator-(TimeType<TRAITS> const t, double const secs)
  { return seconds_before(t, secs); }
template<class TRAITS> inline int operator-(TimeType<TRAITS> const t1, TimeType<TRAITS> const t0)
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
}  // namespace cron

