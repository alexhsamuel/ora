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

