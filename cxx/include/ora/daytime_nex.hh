/*
 * Safe (non-throwing) versions of daytime functions.
 *
 * All functions in `ora::daytime::safe` are `noexcept`; they return accept
 * invalid arguments, and return invalid value on failure.
 */

#pragma once

#include "ora/lib/math.hh"

#include "ora/types.hh"
#include "ora/daytime_type.hh"

namespace ora {
namespace daytime {
namespace nex {

//------------------------------------------------------------------------------
// Forward declarations
//------------------------------------------------------------------------------

template<class DAYTIME> Daytick get_daytick(DAYTIME) noexcept;

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
      in_range<Offset>(0, offset, DAYTIME::OFFSET_MAX)
    ? DAYTIME::from_offset(offset)
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
    ? DAYTIME::from_daytick(daytick)
    : DAYTIME::INVALID;
}


template<class DAYTIME=Daytime, class FROM>
inline DAYTIME
from_daytime(
  FROM const daytime)
  noexcept
{
  return
      daytime.is_missing() ? DAYTIME::MISSING
    : nex::from_daytick<DAYTIME>(nex::get_daytick(daytime));
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


template<class DAYTIME=Daytime>
inline DAYTIME
from_hmsf(
  double const hmsf)
  noexcept
{
  Hour const hour = hmsf / 10000;
  auto ms = fmod(hmsf, 10000);
  Minute const minute = ms / 100;
  Second const second = fmod(ms, 100);
  return from_hms(hour, minute, second);
}


template<class DAYTIME=Daytime> inline DAYTIME from_hms(HmsDaytime const& hms) noexcept
  { return nex::from_hms<DAYTIME>(hms.hour, hms.minute, hms.second); }

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


template<class DAYTIME=Daytime>
inline DAYTIME
from_iso_daytime(
  std::string const& daytime)
  noexcept
{
  auto hms = parse_iso_daytime(daytime);
  return nex::from_hms<DAYTIME>(hms.hour, hms.minute, hms.second);
}  


//------------------------------------------------------------------------------
// Accessors
//------------------------------------------------------------------------------

template<class DAYTIME>
inline bool
is_valid(
  DAYTIME const daytime)
  noexcept
{
  return daytime.is_valid();
}


template<class DAYTIME=Daytime>
inline typename DAYTIME::Offset
get_offset(
  DAYTIME const daytime)
  noexcept
{
  return daytime.offset_;
}


template<class DAYTIME>
inline Daytick
get_daytick(
  DAYTIME const daytime)
  noexcept
{
  return daytime.is_valid() ? daytime.get_daytick() : DAYTICK_INVALID;
}


template<class DAYTIME>
inline HmsDaytime 
get_hms(
  DAYTIME const daytime)  
  noexcept
{
  if (daytime.is_valid()) {
    auto const offset = get_offset(daytime);
    auto const minutes = offset / (SECS_PER_MIN * DAYTIME::Traits::denominator);
    auto const seconds = offset % (SECS_PER_MIN * DAYTIME::Traits::denominator);
    return {
      (Hour)   (minutes / MINS_PER_HOUR),
      (Minute) (minutes % MINS_PER_HOUR),
      (Second) seconds / DAYTIME::Traits::denominator
    };
  }
  else
    return HmsDaytime{};  // invalid
}


template<class DAYTIME>
inline HmsDaytimePacked
get_hms_packed(
  DAYTIME const daytime)  
  noexcept
{
  if (daytime.is_valid()) {
    auto const offset = get_offset(daytime);
    auto const minutes = offset / (SECS_PER_MIN * DAYTIME::Traits::denominator);
    auto const seconds = offset % (SECS_PER_MIN * DAYTIME::Traits::denominator);
    return {
      (Hour)   (minutes / MINS_PER_HOUR),
      (Minute) (minutes % MINS_PER_HOUR),
      (Second) seconds / DAYTIME::Traits::denominator
    };
  }
  else
    return {};  // invalid
}


template<class DAYTIME>
inline double 
get_ssm(
  DAYTIME const daytime)
  noexcept
{
  return 
      daytime.is_valid()
    ? (double) get_offset(daytime) / DAYTIME::Traits::denominator
    : SSM_INVALID;
}


// For convenience.
template<class DAYTIME> inline Hour get_hour(DAYTIME const daytime) noexcept
  { return nex::get_hms(daytime).hour; }
template<class DAYTIME> inline Minute get_minute(DAYTIME const daytime) noexcept
  { return nex::get_hms(daytime).minute; }
template<class DAYTIME> inline Second get_second(DAYTIME const daytime) noexcept
  { return nex::get_hms(daytime).second; }

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
  if (nex::equal(daytime0, daytime1))
    return false;
  else if (daytime0.is_invalid())
    return true;
  else if (daytime1.is_invalid())
    return false;
  else if (daytime0.is_missing())
    return true;
  else if (daytime1.is_missing())
    return false;
  else {
    assert(daytime0.is_valid());
    assert(daytime1.is_valid());
    return get_offset(daytime0) < get_offset(daytime1);
  }
}


template<class DAYTIME>
inline int
compare(
  DAYTIME const daytime0,
  DAYTIME const daytime1)
  noexcept
{
  return 
      nex::equal(daytime0, daytime1) ? 0 
    : nex::before(daytime0, daytime1) ? -1 
    : 1;
}


//------------------------------------------------------------------------------
// Arithmetic
//------------------------------------------------------------------------------

/*
 * Shifts `daytime` forward by `seconds`.
 *
 * The result is modulo a standard 24-hour day.
 */
template<class DAYTIME>
inline DAYTIME
seconds_after(
  DAYTIME const daytime,
  double const seconds)
{
  return 
      daytime.is_valid()
    ? ora::daytime::seconds_after(daytime, seconds)
    : DAYTIME::INVALID;
}


/*
 * Shifts `daytime` backward by `seconds`.
 *
 * The result is modulo a standard 24-hour day.
 */
template<class DAYTIME>
inline DAYTIME
seconds_before(
  DAYTIME const daytime,
  double const seconds)
{
  return 
      daytime.is_valid()
    ? ora::daytime::seconds_before(daytime, seconds)
    : DAYTIME::INVALID;
}


/*
 * The number of seconds between `daytime0` and `daytime1` on the same day.
 *
 * Assumes both are in a single ordinary 24-hour day.  If `daytime1` is earlier,
 * the result is negative.
 */
template<class DAYTIME>
inline double
seconds_between(
  DAYTIME const daytime0,
  DAYTIME const daytime1)
{
  return 
      daytime0.is_valid() && daytime1.is_valid()
    ? ora::daytime::seconds_between(daytime0, daytime1)
    : std::numeric_limits<double>::quiet_NaN();
}


}  // namespace nex

//------------------------------------------------------------------------------
// Comparison operators
//------------------------------------------------------------------------------

template<class T0, class T1> inline bool operator==(DaytimeTemplate<T0> const d0, DaytimeTemplate<T1> const d1) noexcept
  { return nex::equal(d0, DaytimeTemplate<T0>(d1)); }
template<class T0, class T1> inline bool operator!=(DaytimeTemplate<T0> const d0, DaytimeTemplate<T1> const d1) noexcept
  { return !nex::equal(d0, DaytimeTemplate<T0>(d1)); }
template<class T0, class T1> inline bool operator< (DaytimeTemplate<T0> const d0, DaytimeTemplate<T1> const d1) noexcept
  { return nex::before(d0, DaytimeTemplate<T0>(d1)); }
template<class T0, class T1> inline bool operator> (DaytimeTemplate<T0> const d0, DaytimeTemplate<T1> const d1) noexcept
  { return nex::before(DaytimeTemplate<T0>(d1), d0); }
template<class T0, class T1> inline bool operator<=(DaytimeTemplate<T0> const d0, DaytimeTemplate<T1> const d1) noexcept
  { return !nex::before(DaytimeTemplate<T0>(d1), d0); }
template<class T0, class T1> inline bool operator>=(DaytimeTemplate<T0> const d0, DaytimeTemplate<T1> const d1) noexcept
  { return !nex::before(d0, DaytimeTemplate<T0>(d1)); }

//------------------------------------------------------------------------------

}  // namespace daytime
}  // namespace ora

