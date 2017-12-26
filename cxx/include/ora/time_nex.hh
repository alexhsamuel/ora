#pragma once

#include "ora/lib/math.hh"
#include "ora/time_math.hh"
#include "ora/time_type.hh"

namespace ora {
namespace time {
namespace nex {

//------------------------------------------------------------------------------

template<class TIME=Time>
inline TIME
from_offset(
  typename TIME::Offset const offset)
  noexcept
{
  return 
      in_range(TIME::Traits::min, offset, TIME::Traits::max)
    ? TIME::from_offset(offset)
    : TIME::INVALID;
}


template<class TIME=Time>
inline TIME
from_timespec(
  timespec const ts)
{
  return nex::from_offset(timespec_to_offset<TIME>(ts));
}


template<class TIME>
inline EpochTime
get_epoch_time(
  TIME const time)
  noexcept
{
  return
      time.is_valid() 
    ? convert_offset(
        time.get_offset(), TIME::DENOMINATOR, TIME::BASE,
        1, DATENUM_UNIX_EPOCH)
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

}  // namespace time
}  // namespace ora

