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
#include "aslib/math.hh"
#include "aslib/printable.hh"
#include "cron/date.hh"
#include "cron/daytime.hh"
#include "cron/time_math.hh"
#include "cron/time_type.hh"

namespace cron {
namespace time {

//------------------------------------------------------------------------------
// Factory functions
//------------------------------------------------------------------------------

// Synonyms for static factory methods.
template<class TIME=Time> inline TIME from_offset(typename TIME::Offset const o)
  { return TIME::from_offset(o); }

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
    ? from_offset<TIME>(timespec_to_offset<TIME>(ts)) 
    : TIME::INVALID;
}


//------------------------------------------------------------------------------
// Comparisons
//------------------------------------------------------------------------------

template<class TIME>
inline int
compare(
  TIME const time0,
  TIME const time1)
{
  ensure_valid(time0);
  ensure_valid(time1);
  return aslib::compare<typename TIME::Offset>(time0.get_offset(), time1.get_offset());
}


template<class TIME>
inline bool
equal(
  TIME const time0,
  TIME const time1)
{
  ensure_valid(time0);
  ensure_valid(time1);
  return time0.get_offset() == time1.get_offset();
}


template<class TIME>
inline bool
before(
  TIME const time0,
  TIME const time1)
{
  ensure_valid(time0);
  ensure_valid(time1);
  return time0.get_offset() < time1.get_offset();
}


//------------------------------------------------------------------------------
// Arithemtic with seconds
//------------------------------------------------------------------------------

/*
 * Shifts `time` forward by `seconds`.
 */
template<class TIME>
inline TIME
seconds_after(
  TIME const time,
  double const seconds)
{
  using Offset = typename TIME::Offset;
  ensure_valid(time);
  // FIXME: Check for overflow.
  return from_offset<TIME>(
    time.get_offset() + (Offset) round(seconds * TIME::DENOMINATOR));
}


/*
 * Shifts `time` backward by `seconds`.
 */
template<class TIME>
inline TIME
seconds_before(
  TIME const time,
  double const seconds)
{
  using Offset = typename TIME::Offset;
  ensure_valid(time);
  // FIXME: Check for overflow.
  return from_offset<TIME>(
    time.get_offset() - (Offset) round(seconds * TIME::DENOMINATOR));
}


/*
 * The number of seconds between `time0` and `time1.
 */
template<class TIME>
inline double
seconds_between(
  TIME const time0,
  TIME const time1)
{
  ensure_valid(time0);
  ensure_valid(time1);
  return ((double) time1.get_offset() - time0.get_offset()) / TIME::DENOMINATOR;
}


//------------------------------------------------------------------------------
// Addition and subtraction
//------------------------------------------------------------------------------

template<class TRAITS> inline TimeType<TRAITS> operator+(TimeType<TRAITS> const t, double const secs)
  { return seconds_after(t, secs); }
template<class TRAITS> inline TimeType<TRAITS> operator-(TimeType<TRAITS> const t, double const secs)
  { return seconds_before(t, secs); }
template<class TRAITS> inline int operator-(TimeType<TRAITS> const t1, TimeType<TRAITS> const t0)
  { return seconds_between(t0, t1); } 

template<class TRAITS> inline TimeType<TRAITS> operator+=(TimeType<TRAITS>& t, int const secs) 
  { return t = t + secs; }
template<class TRAITS> inline TimeType<TRAITS> operator++(TimeType<TRAITS>& t) 
  { return t = t + 1; }
template<class TRAITS> inline TimeType<TRAITS> operator++(TimeType<TRAITS>& t, int /* tag */) 
  { auto old = t; t = t + 1; return old; }
template<class TRAITS> inline TimeType<TRAITS> operator-=(TimeType<TRAITS>& t, int const secs) 
  { return t = t - secs; }
template<class TRAITS> inline TimeType<TRAITS> operator--(TimeType<TRAITS>& t) 
  { return t = t - 1; }
template<class TRAITS> inline TimeType<TRAITS> operator--(TimeType<TRAITS>& t, int /* tag */) 
  { auto old = t; t = t - 1; return old; }

//------------------------------------------------------------------------------

}  // namespace time
}  // namespace cron

