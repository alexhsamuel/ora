#pragma once

#include "aslib/math.hh"

#include "cron/types.hh"
#include "cron/daytime_type.hh"

namespace cron {
namespace daytime {
namespace safe {

//------------------------------------------------------------------------------
// Factory functions
//------------------------------------------------------------------------------

template<class DAYTIME=Daytime>
inline DAYTIME
from_offset(
  typename DAYTIME::Offset const offset)
  noexcept
{
  using Offset = typename DAYTIME::Offset;
  return 
      in_range<Offset>(0, offset, DAYTIME::MAX_OFFSET)
    ? DAYTIME(offset)
    : DAYTIME::INVALID;
}


template<class DAYTIME=Daytime>
inline DAYTIME
from_daytick(
  Daytick const daytick)
  noexcept
{
  return 
      daytick_is_valid(daytick)
    ? DAYTIME(
        rescale_int<Daytick, DAYTICK_PER_SEC, DAYTIME::DENOMINATOR>(daytick))
    : DAYTIME::INVALID;
}


template<class DAYTIME=Daytime>
inline DAYTIME
from_hms(
  Hour const hour,
  Minute const minute,
  Second const second=0)
  noexcept
{
  using Offset = typename DAYTIME::Offset;
  return 
      hms_is_valid(hour, minute, second)
    ? from_offset<DAYTIME>(
        (hour * SECS_PER_HOUR + minute * SECS_PER_MIN) * DAYTIME::DENOMINATOR
      + (Offset) (second * DAYTIME::DENOMINATOR))
    : DAYTIME::INVALID;
}


template<class DAYTIME=Daytime> inline DAYTIME from_hms(HmsDaytime const& hms)
  { return safe::from_hms<DAYTIME>(hms.hour, hms.minute, hms.second); }

template<class DAYTIME=Daytime>
inline DAYTIME
from_ssm(
  Ssm const ssm)
  noexcept
{
  return 
      ssm_is_valid(ssm)
    ? from_offset<DAYTIME>(round(ssm * DAYTIME::DENOMINATOR))
    : DAYTIME::INVALID;
}


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

