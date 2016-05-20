#pragma once

#include "cron/types.hh"

namespace cron {
namespace daytime {

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

template<class TRAITS>
inline DaytimeTemplate<TRAITS>
operator+(
  DaytimeTemplate<TRAITS> const daytime,
  double const shift)
{
  using Daytime = DaytimeTemplate<TRAITS>;

  if (daytime.is_invalid() || daytime.is_missing())
    return daytime;
  else {
    auto offset = daytime.get_offset();
    offset += round(shift * Daytime::DENOMINATOR);
    return Daytime::from_offset(offset % (SECS_PER_DAY * Daytime::DENOMINATOR));
  }
}


template<class TRAITS>
inline DaytimeTemplate<TRAITS>
operator-(
  DaytimeTemplate<TRAITS> const daytime,
  double shift)
{
  using Daytime = DaytimeTemplate<TRAITS>;

  if (shift > SECS_PER_DAY)
    shift = fmod(shift, SECS_PER_DAY);

  if (daytime.is_invalid() || daytime.is_missing())
    return daytime;
  else {
    auto shift_offset = 
      (typename Daytime::Offset) round(shift * Daytime::DENOMINATOR);
    auto offset = daytime.get_offset();
    // Avoid a negative result.
    if (offset < shift_offset)
      offset += SECS_PER_DAY * Daytime::DENOMINATOR;
    offset -= shift_offset;
    return Daytime::from_offset(offset);
  }
}


//------------------------------------------------------------------------------

}  // namespace daytime
}  // namespace cron

