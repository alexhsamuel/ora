#pragma once

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
  catch (NonexistentLocalTime) {
    // FIXME: Don't catch and rethrow...
    throw NonexistentLocalTime();
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


template<class TIME>
inline LocalDatenumDaytick
to_local_datenum_daytick(
  TIME const time,
  TimeZone const& tz)
{
  using Offset = typename TIME::Offset;

  // Look up the time zone offset for this time.
  auto const tz_offset = tz.get_parts(time).offset;
  // Compute the local offset.
  auto const offset 
    = (Offset) (time.get_offset() + tz_offset * TIME::DENOMINATOR);

  // Establish the date and daytime parts, using division rounded toward -inf
  // and a positive remainder.
  Datenum const datenum   
    =   (int64_t) (offset / TIME::DENOMINATOR) / SECS_PER_DAY 
      + (offset < 0 ? -1 : 0)
      + TIME::BASE;
  Offset const day_offset 
    =   (int64_t) offset % (TIME::DENOMINATOR * SECS_PER_DAY) 
      + (offset < 0 ? TIME::DENOMINATOR * SECS_PER_DAY : 0);
  // FIXME: Not sure the types are right here.
  Daytick const daytick = rescale_int(
    (intmax_t) day_offset, 
    (intmax_t) TIME::DENOMINATOR, (intmax_t) DAYTICK_PER_SEC);

  return {datenum, daytick};
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

