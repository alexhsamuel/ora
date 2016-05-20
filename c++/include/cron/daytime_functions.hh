#pragma once

#include "cron/types.hh"

namespace cron {
namespace daytime {

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

