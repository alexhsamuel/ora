#pragma once

#include <string>

#include "cron/date_functions.hh"
#include "cron/date_type.hh"
#include "cron/daytime_functions.hh"
#include "cron/daytime_type.hh"
#include "cron/exceptions.hh"
#include "cron/time_functions.hh"
#include "cron/time_type.hh"

namespace cron {

//------------------------------------------------------------------------------

template<class TIME=time::Time>
inline TIME
from_local(
  Datenum const         datenum,
  Daytick const         daytick,
  TimeZone const&       time_zone,
  bool const            first=true)
{
  if (! datenum_is_valid(datenum)) 
    throw InvalidDateError();
  if (! daytick_is_valid(daytick))
    throw InvalidDaytimeError();

  return time::from_offset<TIME>(
    time::datenum_daytick_to_offset<typename TIME::Traits>(
      datenum, daytick, time_zone, first));
}


template<class TIME=time::Time, class DATE, class DAYTIME>
inline TIME
from_local(
  DATE const            date,
  DAYTIME const         daytime,
  TimeZone const&       time_zone,
  bool const            first=true)
{
  ensure_valid(date);
  ensure_valid(daytime);
  return TIME::from_offset(
    time::datenum_daytick_to_offset<typename TIME::Traits>(
      date.get_datenum(), daytime.get_daytick(), time_zone, first));
}


template<class TIME=time::Time>
inline TIME
from_local_parts(
  Year const            year,
  Month const           month,
  Day const             day,
  Hour const            hour,
  Minute const          minute,
  Second const          second,
  TimeZone const&       time_zone,
  bool const            first=true)
{
  if (! ymd_is_valid(year, month, day))
    throw InvalidDateError();
  if (! hms_is_valid(hour, minute, second))
    throw InvalidDaytimeError();

  return time::from_offset<TIME>(
    time::datenum_daytick_to_offset<typename TIME::Traits>(
      ymd_to_datenum(year, month, day), 
      hms_to_daytick(hour, minute, second), 
      time_zone, first));
}


template<class TIME>
inline TimeParts 
get_parts(
  TIME const time,
  TimeZone const& time_zone) 
{
  using Offset = typename TIME::Offset;
  static Offset const secs_per_min = TIME::DENOMINATOR * SECS_PER_MIN;

  Datenum datenum;
  Offset daytime_offset;
  TimeZoneParts tz_parts;
  std::tie(datenum, daytime_offset, tz_parts) = split(time, time_zone);
  
  auto const minutes = daytime_offset / secs_per_min;
  return {
    datenum_to_ymd(datenum),
    HmsDaytime{
      (Hour)   (minutes / MINS_PER_HOUR),
      (Minute) (minutes % MINS_PER_HOUR),
      (Second) (daytime_offset % secs_per_min) / TIME::DENOMINATOR,
    },
    tz_parts,
  };
}


// FIXME: Remove?
template<class TIME>
inline TimeParts 
get_parts(
  TIME const time,
  _DisplayTimeZoneTag /* unused */) 
{ 
  return get_parts(time, *get_display_time_zone()); 
}


template<class TIME>
inline Datenum
get_utc_datenum(
  TIME const time)
{
  ensure_valid(time);
  return
      TIME::Traits::base
    + time.get_offset() / SECS_PER_DAY / TIME::Traits::denominator;
}


template<class DATE, class TIME>
inline DATE
get_utc_date(
  TIME const time)
{
  return date::from_datenum<DATE>(get_utc_datenum(time));
}


template<class TIME>
inline Daytick
get_utc_daytick(
  TIME const time)
{
  ensure_valid(time);
  auto const day_offset
    = time.get_offset() % (SECS_PER_DAY * TIME::Traits::denominator);
  return rescale_int<Daytick, TIME::Traits::denominator, DAYTICK_PER_SEC>
    (day_offset);
}


template<class DAYTIME, class TIME>
inline DAYTIME
get_utc_daytime(
  TIME const time)
{
  return daytime::from_daytick<DAYTIME>(get_utc_daytick(time));
}


template<class DATE=date::Date, class DAYTIME=daytime::Daytime, class TIME>
inline LocalTime<DATE, DAYTIME>
to_local(
  TIME const time,
  TimeZone const& tz)
{
  if (time.is_valid()) {
    auto const ldd = to_local_datenum_daytick(time, tz);
    return {
      date::from_datenum<DATE>(ldd.datenum), 
      daytime::from_daytick<DAYTIME>(ldd.daytick),
      ldd.time_zone
    };
  }
  else
    return {};  // invalid
}


// Variants that take a time zone name  ----------------------------------------

template<class TIME=time::Time, class DATE, class DAYTIME>
inline TIME
from_local(
  DATE const            date,
  DAYTIME const         daytime,
  std::string const&    time_zone_name,
  bool const            first=true)
{
  return from_local(date, daytime, *get_time_zone(time_zone_name), first);
}


template<class TIME>
inline TimeParts 
get_parts(
  TIME const time, 
  std::string const& tz_name)
{ 
  return get_parts(time, *get_time_zone(tz_name)); 
}


template<class TIME=time::Time>
inline TIME
from_local_parts(
  Year const            year,
  Month const           month,
  Day const             day,
  Hour const            hour,
  Minute const          minute,
  Second const          second,
  std::string const&    time_zone_name,
  bool const            first=true)
{
  return from_local_parts(
    year, month, day, hour, minute, second, 
    *get_time_zone(time_zone_name), first);
}


template<class DATE=date::Date, class DAYTIME=daytime::Daytime, class TIME>
inline LocalTime<DATE, DAYTIME>
to_local(
  TIME const time,
  std::string const& time_zone_name)
{
  return to_local(time, *get_time_zone(time_zone_name));
}


// UTC variants  ---------------------------------------------------------------

/*
 * Equivalent to `from_local(date, daytime, UTC)`.
 */
template<class TIME=time::Time, class DATE, class DAYTIME>
inline TIME
from_utc(
  DATE const            date,
  DAYTIME const         daytime)
{
  return from_local(date, daytime, UTC);
}


/*
 * Equivalent to `from_local(year, month, ..., UTC)`.
 */
template<class TIME=time::Time>
inline TIME
from_utc_parts(
  Year const            year,
  Month const           month,
  Day const             day,
  Hour const            hour,
  Minute const          minute,
  Second const          second)
{
  return from_local_parts(year, month, day, hour, minute, second, UTC);
}


template<class DATE=date::Date, class DAYTIME=daytime::Daytime, class TIME>
inline LocalTime<DATE, DAYTIME>
to_utc(
  TIME const time)
{
  return to_local(time, UTC);
}


//------------------------------------------------------------------------------

}  // namespace cron

