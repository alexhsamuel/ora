#pragma once

#include "cron/date.hh"
#include "cron/types.hh"

namespace cron {
namespace date {

//------------------------------------------------------------------------------

class DayInterval
{
public:

  using Days = ssize_t;

  constexpr DayInterval(Days days) : days_(days) {}
  constexpr DayInterval operator-() const { return DayInterval(-days_); }
  constexpr DayInterval operator*(Days mult) const { return DayInterval(mult * days_); }

  Datenum shift(Datenum datenum) const { return datenum + days_; }

private:

  Days days_;

};


inline DayInterval
operator*(
  DayInterval::Days mult,
  DayInterval interval)
{
  return interval * mult;
}


template<class TRAITS> 
inline DateTemplate<TRAITS>
operator+(
  DateTemplate<TRAITS> date,
  DayInterval const& interval)
{
  return DateTemplate<TRAITS>::from_datenum(interval.shift(date.get_datenum()));
}


template<class TRAITS> 
inline DateTemplate<TRAITS>
operator+(
  DayInterval const& interval,
  DateTemplate<TRAITS> date)
{
  return date + interval;
}


template<class TRAITS>
inline DateTemplate<TRAITS>
operator-(
  DateTemplate<TRAITS> date,
  DayInterval const& interval)
{
  return date + -interval;
}


DayInterval constexpr DAY = DayInterval(1);


//------------------------------------------------------------------------------

}  // namespace date
}  // namespace cron


