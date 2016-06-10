#pragma once

#include "aslib/math.hh"

#include "cron/types.hh"
#include "cron/daytime_type.hh"

namespace cron {
namespace daytime {
namespace safe {

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// Comparisons
//------------------------------------------------------------------------------

template<class DAYTIME>
inline bool
equal(
  DAYTIME const daytime0,
  DAYTIME const daytime1)
  noexcept
{
  return daytime0.offset_ == daytime1.offset_;
}


template<class DAYTIME>
inline bool
before(
  DAYTIME const daytime0,
  DAYTIME const daytime1)
  noexcept
{
  if (safe::equal(daytime0, daytime1))
    return false;
  else if (daytime0.is_invalid())
    return true;
  else if (daytime1.is_invalid())
    return false;
  else if (daytime0.is_missing())
    return true;
  else if (daytime1.is_missing())
    return false;
  else 
    return daytime0.get_offset() < daytime1.get_offset();
}


template<class DAYTIME>
inline int
compare(
  DAYTIME const daytime0,
  DAYTIME const daytime1)
  noexcept
{
  return 
      safe::equal(daytime0, daytime1) ? 0 
    : safe::before(daytime0, daytime1) ? -1 
    : 1;
}


}  // namespace safe

//------------------------------------------------------------------------------
// Comparison operators
//------------------------------------------------------------------------------

template<class T0, class T1> inline bool operator==(DaytimeTemplate<T0> const d0, DaytimeTemplate<T1> const d1) noexcept
  { return safe::equal(d0, DaytimeTemplate<T0>(d1)); }
template<class T0, class T1> inline bool operator!=(DaytimeTemplate<T0> const d0, DaytimeTemplate<T1> const d1) noexcept
  { return !safe::equal(d0, DaytimeTemplate<T0>(d1)); }
template<class T0, class T1> inline bool operator< (DaytimeTemplate<T0> const d0, DaytimeTemplate<T1> const d1) noexcept
  { return safe::before(d0, DaytimeTemplate<T0>(d1)); }
template<class T0, class T1> inline bool operator> (DaytimeTemplate<T0> const d0, DaytimeTemplate<T1> const d1) noexcept
  { return safe::before(DaytimeTemplate<T0>(d1), d0); }
template<class T0, class T1> inline bool operator<=(DaytimeTemplate<T0> const d0, DaytimeTemplate<T1> const d1) noexcept
  { return !safe::before(DaytimeTemplate<T0>(d1), d0); }
template<class T0, class T1> inline bool operator>=(DaytimeTemplate<T0> const d0, DaytimeTemplate<T1> const d1) noexcept
  { return !safe::before(d0, DaytimeTemplate<T0>(d1)); }

//------------------------------------------------------------------------------

}  // namespace daytime
}  // namespace cron

