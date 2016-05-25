#pragma once

#include "cron/daytime.hh"
#include "cron/time.hh"
#include "cron/types.hh"

namespace cron {

//------------------------------------------------------------------------------

// FIXME: Should we use multiple denominators here?

class TimeInterval
{
public:

  using Dayticks = int64_t;

  constexpr 
  TimeInterval(
    double seconds) 
    : dayticks_(seconds * DAYTICK_PER_SEC) 
  {
  }

  static constexpr TimeInterval
  from_daytick(
    Daytick daytick)
  {
    // FIXME: Check for overflow.
    return TimeInterval((Dayticks) daytick);
  }

  Dayticks get_dayticks() const { return dayticks_; }

  constexpr TimeInterval operator-() const { return TimeInterval::from_daytick(-dayticks_); }
  constexpr TimeInterval operator*(double mult) const { return TimeInterval::from_daytick(mult * dayticks_); }

private:

  constexpr
  TimeInterval(
    Dayticks dayticks)
    : dayticks_(dayticks)
  {
  }

  Dayticks dayticks_;

};


inline TimeInterval
operator*(
  double mult,
  TimeInterval interval)
{
  return interval * mult;
}


template<class TIME>
inline TIME
operator+(
  TIME const time,
  TimeInterval const& interval)
{
  return TIME::from_offset(
    time.get_offset() 
    // FIXME
    + rescale_int((intmax_t) interval.get_dayticks(), (intmax_t) DAYTICK_PER_SEC, (intmax_t) TIME::DENOMINATOR));
}


template<class TIME>
inline TIME
operator+(
  TimeInterval const& interval,
  TIME const time)
{
  return time + interval;
}


TimeInterval constexpr NANOSECOND   = TimeInterval(1.0e-9);
TimeInterval constexpr MICROSECOND  = TimeInterval(1.0e-6);
TimeInterval constexpr MILLISECOND  = TimeInterval(1.0e-3);
TimeInterval constexpr SECOND       = TimeInterval(1.0);
TimeInterval constexpr MINUTE       = TimeInterval((double) SECS_PER_MIN);
TimeInterval constexpr HOUR         = TimeInterval((double) SECS_PER_HOUR);


//------------------------------------------------------------------------------

}  // namespace cron


