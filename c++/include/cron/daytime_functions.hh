#pragma once

#include <cmath>

#include "aslib/math.hh"

#include "cron/types.hh"
#include "cron/daytime_type.hh"

namespace cron {
namespace daytime {

//------------------------------------------------------------------------------
// Factory functions
//------------------------------------------------------------------------------

template<class DAYTIME=Daytime>
inline DAYTIME
from_offset(
  typename DAYTIME::Offset const o)
{
  return DAYTIME::from_offset(o);
}


template<class DAYTIME=Daytime>
inline DAYTIME 
from_daytick(
  Daytick const d)
{ 
  return DAYTIME::from_daytick(d); 
}


/*
 * Creates a daytime from hour, minute, and second components.
 */
template<class DAYTIME=Daytime> 
inline DAYTIME
from_hms(
  Hour const hour,
  Minute const minute,
  Second const second=0)
{ 
  using Offset = typename DAYTIME::Offset;
  if (hms_is_valid(hour, minute, second)) 
    return from_offset<DAYTIME>(
        (hour * SECS_PER_HOUR + minute * SECS_PER_MIN) * DAYTIME::DENOMINATOR
      + (Offset) (second * DAYTIME::DENOMINATOR));
  else
    throw InvalidDaytimeError();
}


template<class DAYTIME=Daytime> inline DAYTIME from_hms(HmsDaytime const& hms)
  { return from_hms<DAYTIME>(hms.hour, hms.minute, hms.second); }

/*
 * Creates a daytime from SSM (seconds since midnight).
 */
template<class DAYTIME=Daytime>
inline DAYTIME
from_ssm(
  Ssm const ssm)
{ 
  if (ssm_is_valid(ssm))
    return from_offset<DAYTIME>(round(ssm * DAYTIME::DENOMINATOR));
  else
    throw InvalidDaytimeError();
}


//------------------------------------------------------------------------------
// Accessors
//------------------------------------------------------------------------------

template<class DAYTIME>
inline double 
get_ssm(
  DAYTIME const daytime)
{
  ensure_valid(daytime);
  return (double) daytime.get_offset() / DAYTIME::Traits::denominator; 
}


template<class DAYTIME>
inline HmsDaytime 
get_hms(
  DAYTIME const daytime)  
{
  ensure_valid(daytime);
  auto const offset = daytime.get_offset();
  auto const minutes = offset / (SECS_PER_MIN * DAYTIME::Traits::denominator);
  auto const seconds = offset % (SECS_PER_MIN * DAYTIME::Traits::denominator);
  return {
    (Hour)   (minutes / MINS_PER_HOUR),
    (Minute) (minutes % MINS_PER_HOUR),
    (Second) seconds / DAYTIME::Traits::denominator
  };
}


// For convenience.
template<class DAYTIME> inline Hour get_hour(DAYTIME const daytime)
  { return get_hms(daytime).hour; }
template<class DAYTIME> inline Minute get_minute(DAYTIME const daytime)
  { return get_hms(daytime).minute; }
template<class DAYTIME> inline Second get_second(DAYTIME const daytime)
  { return get_hms(daytime).second; }

//------------------------------------------------------------------------------
// Comparisons
//------------------------------------------------------------------------------

template<class DAYTIME>
inline int
compare(
  DAYTIME const daytime0,
  DAYTIME const daytime1)
{
  ensure_valid(daytime0);
  ensure_valid(daytime1);
  return aslib::compare(daytime0.get_offset(), daytime1.get_offset());
}


template<class DAYTIME>
inline bool
equal(
  DAYTIME const daytime0,
  DAYTIME const daytime1)
{
  ensure_valid(daytime0);
  ensure_valid(daytime1);
  return daytime0.get_offset() == daytime1.get_offset();
}


template<class DAYTIME>
inline bool
before(
  DAYTIME const daytime0,
  DAYTIME const daytime1)
{
  ensure_valid(daytime0);
  ensure_valid(daytime1);
  return daytime0.get_offset() < daytime1.get_offset();
}


//------------------------------------------------------------------------------
// Arithemtic with seconds
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
  using Offset = typename DAYTIME::Offset;

  ensure_valid(daytime);
  double offset = daytime.get_offset() + seconds * DAYTIME::DENOMINATOR;

  double const day_offset = DAYTIME::OFFSET_END;
  if (offset >= 0)
    offset = fmod(offset, day_offset);
  else
    offset = fmod(offset, day_offset) + day_offset;
  return from_offset<DAYTIME>((Offset) round(offset));
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
  return seconds_after(daytime, -seconds);
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
  ensure_valid(daytime0);
  ensure_valid(daytime1);

  auto const off0 = daytime0.get_offset();
  auto const off1 = daytime1.get_offset();
  return 
    (off1 >= off0 ? (double) (off1 - off0) : -(double) (off0 - off1)) 
    / DAYTIME::DENOMINATOR;
}


//------------------------------------------------------------------------------
// Addition and subtraction
//------------------------------------------------------------------------------

template<class DAYTIME> inline DAYTIME operator+(DAYTIME const d, double const secs)
  { return seconds_after(d, secs); }
template<class DAYTIME> inline DAYTIME operator-(DAYTIME const d, double const secs)
  { return seconds_before(d, secs); }
template<class DAYTIME> inline int operator-(DAYTIME const d1, DAYTIME const d0)
  { return seconds_between(d0, d1); } 

template<class DAYTIME> inline DAYTIME operator+=(DAYTIME& d, int const secs) 
  { return d = d + secs; }
template<class DAYTIME> inline DAYTIME operator++(DAYTIME& d) 
  { return d = d + 1; }
template<class DAYTIME> inline DAYTIME operator++(DAYTIME& d, int /* tag */) 
  { auto old = d; d = d + 1; return old; }
template<class DAYTIME> inline DAYTIME operator-=(DAYTIME& d, int const secs) 
  { return d = d - secs; }
template<class DAYTIME> inline DAYTIME operator--(DAYTIME& d) 
  { return d = d - 1; }
template<class DAYTIME> inline DAYTIME operator--(DAYTIME& d, int /* tag */) 
  { auto old = d; d = d - 1; return old; }

//------------------------------------------------------------------------------

}  // namespace daytime
}  // namespace cron

//------------------------------------------------------------------------------
// Namespace imports
//------------------------------------------------------------------------------

namespace cron {

using daytime::from_hms;
using daytime::from_ssm;

}  // namespace cron

