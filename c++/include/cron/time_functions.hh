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
  return convert_offset(
    time.get_offset(), TIME::DENOMINATOR, TIME::BASE,
    1, DATENUM_UNIX_EPOCH);
}


template<class TIME=Time>
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

