#pragma once

#include "cron/types.hh"

namespace cron {

//------------------------------------------------------------------------------

/*
 * Not aware of leap hours.
 */
inline constexpr bool
hms_is_valid(
  Hour const hour,
  Minute const minute,
  Second const second)
{
  return 
       hour_is_valid(hour)
    && minute_is_valid(minute)
    && second_is_valid(second);
}


inline constexpr Daytick
hms_to_daytick(
  Hour const hour,
  Minute const minute,
  Second const second)
{
  return 
      (hour * SECS_PER_HOUR + minute * SECS_PER_MIN) * DAYTICK_PER_SEC
    + second * DAYTICK_PER_SEC;
}


//------------------------------------------------------------------------------

}  // namespace cron

