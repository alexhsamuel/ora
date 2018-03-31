#pragma once

#include "ora/time_math.hh"
#include "ora/types.hh"

namespace ora {
namespace nex {

//------------------------------------------------------------------------------

template<class TIME=time::Time>
inline TIME
from_local(
  Datenum const         datenum,
  Daytick const         daytick,
  TimeZone const&       time_zone,
  bool const            first=true)
{
  if (datenum_is_valid(datenum) && daytick_is_valid(daytick))
    return time::nex::from_offset<TIME>(
      time::datenum_daytick_to_offset<typename TIME::Traits>(
        datenum, daytick, time_zone, first));
  else
    return TIME::INVALID;
}


template<class TIME>
LocalDatenumDaytick
to_local_datenum_daytick(
  TIME const time,
  TimeZone const& time_zone)
{
  return
      time.is_valid()
    ? ora::time::to_local_datenum_daytick(time, time_zone)
    : LocalDatenumDaytick{};
}


//------------------------------------------------------------------------------

}  // namespace nex
}  // namespace ora

