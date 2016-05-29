#pragma once

#include <time.h>

#ifdef __MACH__
#include <mach/clock.h>
#include <mach/mach.h>
// I don't f***ing believe it.
#undef TRUE
#undef FALSE
#endif

#include "aslib/exc.hh"
#include "aslib/printable.hh"
#include "cron/date.hh"
#include "cron/daytime.hh"
#include "cron/time_math.hh"
#include "cron/time_type.hh"

namespace cron {
namespace time {

//------------------------------------------------------------------------------

/*
 * Returns the closest UNIX epoch time.
 *
 * Returns the rounded (signed) number of seconds since 1970-01-01T00:00:00Z.
 */
template<class TIME>
inline int64_t
get_epoch_time(
  TIME const time)
{
  return Unix64Time(time).get_offset();
}


template<class TIME>
inline TimeParts 
get_parts(
  TIME const time,
  TimeZone const& tz) 
{
  using Offset = typename TIME::Offset;
  static Offset const secs_per_day = TIME::DENOMINATOR * SECS_PER_DAY;
  static Offset const secs_per_min = TIME::DENOMINATOR * SECS_PER_MIN;

  TimeParts parts;

  // Look up the time zone.
  parts.time_zone = tz.get_parts(time);
  Offset const offset 
    = time.get_offset() + parts.time_zone.offset * TIME::DENOMINATOR;

  // Establish the date and daytime parts, using division rounded toward -inf
  // and a positive remainder.
  Datenum const datenum   
    =   (int64_t) (offset / secs_per_day)
      + (offset < 0 ? -1 : 0)
      + TIME::BASE;
  parts.date = datenum_to_parts(datenum);

  auto const day_offset 
    = offset % secs_per_day + (offset < 0 ? secs_per_day : 0);
  parts.daytime.second  
    = (Second) (day_offset % secs_per_min) / TIME::DENOMINATOR;
  Offset const minutes  = day_offset / secs_per_min;
  parts.daytime.minute  = minutes % MINS_PER_HOUR;
  parts.daytime.hour    = minutes / MINS_PER_HOUR;

  return parts;
}


template<class TIME>
inline TimeParts 
get_parts(
  TIME const time, 
  std::string const& tz_name)
{ 
  return get_parts(time, *get_time_zone(tz_name)); 
}


template<class TIME>
inline TimeParts 
get_parts(
  TIME const time,
  _DisplayTimeZoneTag /* unused */) 
{ 
  return get_parts(time, *get_display_time_zone()); 
}


template<class DATE, class TIME>
inline DATE
get_utc_date(
  TIME const time)
{
  return date::from_datenum<DATE>(get_utc_datenum(time));
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


template<class TIME>
inline TIME
now()
{
  struct timespec ts;
  bool success;

#ifdef __MACH__
  clock_serv_t cclock;
  mach_timespec_t mts;
  success = 
       host_get_clock_service(mach_host_self(), CALENDAR_CLOCK, &cclock) == 0
    && clock_get_time(cclock, &mts) == 0;
  mach_port_deallocate(mach_task_self(), cclock);
  ts.tv_sec = mts.tv_sec;
  ts.tv_nsec = mts.tv_nsec;
#else
  success = clock_gettime(CLOCK_REALTIME, &ts) == 0;
#endif

  return 
      success
    ? TIME::from_offset(cron::time::timespec_to_offset<TIME>(ts)) 
    : TIME::INVALID;
}


template<class DATE, class DAYTIME, class TIME>
inline LocalTime<DATE, DAYTIME>
to_local(
  TIME const time,
  TimeZone const& tz)
{
  if (time.is_valid()) {
    auto dd = to_local_datenum_daytick(time, tz);
    return {
      date::from_datenum<DATE>(dd.datenum), 
      TIME::from_daytick(dd.daytick)
    };
  }
  else
    return {};  // invalid
}


//------------------------------------------------------------------------------

template<class TRAITS>
inline TimeTemplate<TRAITS>
operator+(
  TimeTemplate<TRAITS> const time,
  double const shift)
{
  using Time = TimeTemplate<TRAITS>;
  using Offset = typename Time::Offset;
  return
      time.is_invalid() || time.is_missing() ? time
    : Time::from_offset(
        time.get_offset() + (Offset) (shift * Time::DENOMINATOR));
}


template<class TRAITS>
inline TimeTemplate<TRAITS>
operator-(
  TimeTemplate<TRAITS> const time,
  double const shift)
{
  using Time = TimeTemplate<TRAITS>;
  using Offset = typename Time::Offset;
  return
      time.is_invalid() || time.is_missing() ? time
    : Time::from_offset(
        time.get_offset() - (Offset) (shift * Time::DENOMINATOR));
}


template<class TRAITS>
inline double
operator-(
  TimeTemplate<TRAITS> const time0,
  TimeTemplate<TRAITS> const time1)
{
  using Time = TimeTemplate<TRAITS>;

  if (time0.is_valid() && time1.is_valid())
    return 
        (double) time0.get_offset() / Time::DENOMINATOR
      - (double) time1.get_offset() / Time::DENOMINATOR;
  else
    throw cron::ValueError("can't subtract invalid times");
}


//------------------------------------------------------------------------------

}  // namespace time
}  // namespace cron

