#pragma once

#include "cron/types.hh"

namespace cron {

//------------------------------------------------------------------------------

extern inline bool
daytick_is_valid(
  Daytick const daytick)
{
  return in_range(DAYTICK_MIN, daytick, DAYTICK_MAX);
}


/**
 * Not aware of leap hours.
 */
extern inline bool
hms_is_valid(
  Hour hour,
  Minute minute,
  Second second)
{
  return 
       hour_is_valid(hour)
    && minute_is_valid(minute)
    && second_is_valid(second);
}


extern inline Daytick
hms_to_daytick(
  Hour hour,
  Minute minute,
  Second second)
{
  return 
      (hour * SECS_PER_HOUR + minute * SECS_PER_MIN) * DAYTICK_PER_SEC
    + second * DAYTICK_PER_SEC;
}


//------------------------------------------------------------------------------

}  // namespace cron


