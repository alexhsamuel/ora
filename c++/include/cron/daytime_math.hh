#pragma once

#include "cron/types.hh"

namespace cron {

//------------------------------------------------------------------------------

// FIXME: Merge with get_hms();
inline HmsDaytime
daytick_to_hms(
  Daytick const daytick)
{
  auto const minutes = daytick / (SECS_PER_MIN * DAYTICK_PER_SEC);
  auto const seconds = daytick % (SECS_PER_MIN * DAYTICK_PER_SEC);
  return {
    (Hour)   (minutes / MINS_PER_HOUR),
    (Minute) (minutes % MINS_PER_HOUR),
    (Second) seconds / DAYTICK_PER_SEC 
  };
}


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


/*
 * Parses an ISO-8601 extended daytime ("HH:MM:SS" format) into parts.
 */
extern HmsDaytime parse_iso_daytime(std::string const&) noexcept;

//------------------------------------------------------------------------------

}  // namespace cron

