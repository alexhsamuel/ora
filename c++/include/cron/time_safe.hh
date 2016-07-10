#pragma once

#include "cron/types.hh"
#include "cron/time_type.hh"

namespace cron {
namespace time {
namespace safe {

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// Comparisons
//------------------------------------------------------------------------------

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
  if (safe::equal(time0, time1))
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
      safe::equal(time0, time1) ? 0
    : safe::before(time0, time1) ? -1
    : 1;
}


}  // namespace safe

//------------------------------------------------------------------------------
// Comparison operators
//------------------------------------------------------------------------------

template<class T0, class T1> inline bool operator==(TimeType<T0> const t0, TimeType<T1> const t1) noexcept
  { return safe::equal(t0, TimeType<T0>(t1)); }
template<class T0, class T1> inline bool operator!=(TimeType<T0> const t0, TimeType<T1> const t1) noexcept
  { return !safe::equal(t0, TimeType<T0>(t1)); }
template<class T0, class T1> inline bool operator< (TimeType<T0> const t0, TimeType<T1> const t1) noexcept
  { return safe::before(t0, TimeType<T0>(t1)); }
template<class T0, class T1> inline bool operator> (TimeType<T0> const t0, TimeType<T1> const t1) noexcept
  { return safe::before(TimeType<T0>(t1), t0); }
template<class T0, class T1> inline bool operator<=(TimeType<T0> const t0, TimeType<T1> const t1) noexcept
  { return !safe::before(TimeType<T0>(t1), t0); }
template<class T0, class T1> inline bool operator>=(TimeType<T0> const t0, TimeType<T1> const t1) noexcept
  { return !safe::before(t0, TimeType<T0>(t1)); }

}  // namespace time
}  // namespace cron

//------------------------------------------------------------------------------
// Namespace imports
//------------------------------------------------------------------------------

namespace cron {
namespace safe {

using namespace time::safe;

}  // namespace safe
}  // namespace cron

