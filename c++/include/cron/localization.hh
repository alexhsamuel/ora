#pragma once

#include <string>

#include "cron/date.hh"
#include "cron/daytime.hh"
#include "cron/time.hh"
#include "cron/time_zone.hh"

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

  return TIME::from_offset(
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


template<class TIME=time::Time>
inline TIME
from_local(
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

  return TIME::from_offset(
    time::datenum_daytick_to_offset<typename TIME::Traits>(
      ymd_to_datenum(year, month, day), 
      hms_to_daytick(hour, minute, second), 
      time_zone, first));
}


template<class TIME=time::Time>
inline TIME
from_local(
  Year const            year,
  Month const           month,
  Day const             day,
  Hour const            hour,
  Minute const          minute,
  Second const          second,
  std::string const&    time_zone_name,
  bool const            first=true)
{
  return from_local(
    year, month, day, hour, minute, second, 
    *get_time_zone(time_zone_name), first);
}


//------------------------------------------------------------------------------

}  // namespace cron

