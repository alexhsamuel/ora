#pragma once

#include "ora/time_math.hh"
#include "ora/types.hh"

namespace ora {
namespace nex {

//------------------------------------------------------------------------------

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

