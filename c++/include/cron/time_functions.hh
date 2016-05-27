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
#include "cron/time_zone.hh"

namespace cron {
namespace time {

//------------------------------------------------------------------------------

template<class TIME>
inline TIME
from_local(
  Datenum const datenum,
  Daytick const daytick,
  TimeZone const& time_zone,
  bool const first=true)
{
  // FIXME: Move the logic here, instead of delegating.
  return {datenum, daytick, time_zone, first};
}


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
    ? TIME::from_offset(cron::time::timespec_to_offset<Time>(ts)) 
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

