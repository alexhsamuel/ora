#pragma once

#include <time.h>

#ifdef __MACH__
#include <mach/clock.h>
#include <mach/mach.h>
// I don't f***ing believe it.
#undef TRUE
#undef FALSE
#endif

#include "cron/time_zone.hh"
#include "cron/types.hh"

namespace cron {
namespace time { 

//------------------------------------------------------------------------------
// Helper functions
//------------------------------------------------------------------------------

template<class OFFSET0, class OFFSET1>
inline OFFSET1
convert_offset(
  OFFSET0   const offset0,
  OFFSET0   const denominator0,
  Datenum   const base0,
  OFFSET1   const denominator1,
  Datenum   const base1)
{
  return
      rescale_int(offset0, denominator0, denominator1)
    + ((long) base0 - base1) * SECS_PER_DAY * denominator1;
}


template<class TRAITS>
inline typename TRAITS::Offset 
datenum_daytick_to_offset(
  Datenum const datenum,
  Daytick const daytick,
  TimeZone const& tz,
  bool const first)
{
  using Offset = typename TRAITS::Offset;
  static auto constexpr denominator = TRAITS::denominator;
  static auto constexpr base = TRAITS::base;
  // The datenum of the day containing the minimum time.
  static auto constexpr min_datenum 
    = (Datenum) (TRAITS::base + TRAITS::min / (Offset) SECS_PER_DAY);

  Offset tz_offset;
  try {
    tz_offset = tz.get_parts_local(datenum, daytick, first).offset;
  }
  catch (NonexistentDateDaytime) {
    // FIXME: Don't catch and rethrow...
    throw NonexistentDateDaytime();
  }

  Offset date_diff = (int64_t) datenum - base;
  Offset daytime_offset
    =   rescale_int(daytick, DAYTICK_PER_SEC, denominator) 
      - denominator * tz_offset;

  // Special case: if the time occurs on the first representable date, but
  // midnight of that date is not representable, we'd overflow if we computed
  // the midnight offset first and then added the daytime.  To get around this,
  // shift the date forward a day and nock a day off the daytime offset.
  if (   TRAITS::min < 0
      && TRAITS::min % SECS_PER_DAY != 0
      && daytime_offset > 0
      && datenum < min_datenum) {
    ++date_diff;
    daytime_offset -= SECS_PER_DAY * denominator;
  }
    
  // Compute the offset for midnight on the date, then add the offset for the
  // daytime, checking for overflows.
  Offset offset;
  if (   mul_overflow(denominator * SECS_PER_DAY, date_diff, offset)
      || add_overflow(offset, daytime_offset, offset))
    throw TimeRangeError();
  else
    return offset;
}


inline timespec
now_timespec()
{
  timespec ts;

#ifdef __MACH__
  clock_serv_t cclock;
  mach_timespec_t mts;
  // FIXME: Should we keep the clock service around?
  bool const success = 
       host_get_clock_service(mach_host_self(), CALENDAR_CLOCK, &cclock) == 0
    && clock_get_time(cclock, &mts) == 0;
  mach_port_deallocate(mach_task_self(), cclock);
  if (success) {
    ts.tv_sec = mts.tv_sec;
    ts.tv_nsec = mts.tv_nsec;
  }
  else 
    ts.tv_nsec = -1;
#else
  if (clock_gettime(CLOCK_REALTIME, &ts) != 0)
    ts.tv_nsec = -1;
#endif

  return ts;
}


/*
 * Splits a time into localized date and daytime parts.
 *
 * For <time> in <time_zone>, returns the local datenum, the residual daytime
 * offset, and the time zone state.
 */
template<class TIME>
inline std::tuple<Datenum, typename TIME::Offset, TimeZoneParts>
split(
  TIME const time,
  TimeZone const& time_zone)
{
  using Offset = typename TIME::Offset;
  static Offset const secs_per_day = TIME::DENOMINATOR * SECS_PER_DAY;

  // Look up the time zone offset for this time.
  auto const tz = time_zone.get_parts(time);
  // Compute the local offset.
  Offset const offset = time.get_offset() + tz.offset * TIME::DENOMINATOR;
  // Establish the date and daytime parts, using division rounded toward -inf
  // and a positive remainder.
  auto const div = sgndiv(offset, secs_per_day);
  // We may need signed addition to compute the datenum.
  Datenum const datenum = (int64_t) div.quot + TIME::BASE; 

  return std::make_tuple(datenum, div.rem, tz);
}


template<class TIME>
inline LocalDatenumDaytick
to_local_datenum_daytick(
  TIME const time,
  TimeZone const& time_zone)
{
  auto parts = split(time, time_zone);

  // FIXME: Not sure the types are right here.
  Daytick const daytick = rescale_int(
    std::get<1>(parts), TIME::DENOMINATOR, DAYTICK_PER_SEC);

  return {std::get<0>(parts), daytick, std::get<2>(parts)};
}


/* 
 * Converts the time in a a 'struct timespec' to an offset for 'TIME'.
 */
template<class TIME>
inline typename TIME::Offset
timespec_to_offset(
  struct timespec const& ts)
{
  using Offset = typename TIME::Offset;
  return 
      ((Offset) (DATENUM_UNIX_EPOCH - TIME::BASE) * SECS_PER_DAY + ts.tv_sec) 
    * TIME::DENOMINATOR
    + rescale_int<Offset, (Offset) 1000000000, TIME::DENOMINATOR>(ts.tv_nsec);
}


//------------------------------------------------------------------------------

}  // namespace time
}  // namespace cron

