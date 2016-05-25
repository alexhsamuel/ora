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

//------------------------------------------------------------------------------

template<typename TRAITS>
inline TimeTemplate<TRAITS>
operator+(
  TimeTemplate<TRAITS> time,
  double shift)
{
  using Time = TimeTemplate<TRAITS>;
  using Offset = typename Time::Offset;
  return
      time.is_invalid() || time.is_missing() ? time
    : Time::from_offset(
        time.get_offset() + (Offset) (shift * Time::DENOMINATOR));
}


template<typename TRAITS>
inline TimeTemplate<TRAITS>
operator-(
  TimeTemplate<TRAITS> time,
  double shift)
{
  using Time = TimeTemplate<TRAITS>;
  using Offset = typename Time::Offset;
  return
      time.is_invalid() || time.is_missing() ? time
    : Time::from_offset(
        time.get_offset() - (Offset) (shift * Time::DENOMINATOR));
}


template<typename TRAITS>
inline double
operator-(
  TimeTemplate<TRAITS> time0,
  TimeTemplate<TRAITS> time1)
{
  using Time = TimeTemplate<TRAITS>;

  if (time0.is_valid() && time1.is_valid())
    return 
        (double) time0.get_offset() / Time::DENOMINATOR
      - (double) time1.get_offset() / Time::DENOMINATOR;
  else if (Time::USE_INVALID)
    // FIXME: What do we do with invalid/missing values?
    return 0;
  else
    throw cron::ValueError("can't subtract invalid times");
}


template<typename TIME>
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


template<typename TIME>
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
    ? TIME::from_offset(cron::timespec_to_offset<Time>(ts)) 
    : TIME::INVALID;
}


template<typename TIME, typename DATE, typename DAYTIME>
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
    // FIXME: LocalTime::INVALID?
    return {DATE::INVALID, DAYTIME::INVALID};
}


//------------------------------------------------------------------------------

}  // namespace cron

