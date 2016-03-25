#pragma once

#include "cron/date.hh"
#include "cron/types.hh"

namespace alxs {
namespace cron {

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

class MonthInterval
{
public:

  using Months = ssize_t;

  constexpr MonthInterval(Months months) : months_(months) {}
  constexpr MonthInterval operator-() const { return MonthInterval(-months_); }
  constexpr MonthInterval operator*(Months mult) const { return MonthInterval(mult * months_); }

  Datenum 
  shift(
    Datenum datenum) 
    const
  {
    if (months_ == 0)
      return datenum;
    else {
      DateParts parts = datenum_to_parts(datenum);
      Months const month = parts.month + months_;
      if (month >= 0) {
        parts.year += month / 12;
        parts.month = month % 12;
      }
      else {
        parts.year += month / 12 - 1;
        parts.month = month % 12 + 12;
      }
      parts.day = std::min(parts.day, (Day) (days_per_month(parts.year, parts.month) - 1));
      assert(ymd_is_valid(parts.year, parts.month, parts.day));
      return ymd_to_datenum(parts.year, parts.month, parts.day);
    }
  }


private:

  Months months_;

};


inline MonthInterval
operator*(
  MonthInterval::Months mult,
  MonthInterval interval)
{
  return interval * mult;
}


template<class TRAITS> 
inline DateTemplate<TRAITS>
operator+(
  DateTemplate<TRAITS> date,
  MonthInterval const& interval)
{
  return DateTemplate<TRAITS>::from_datenum(interval.shift(date.get_datenum()));
}


template<class TRAITS> 
inline DateTemplate<TRAITS>
operator+(
  MonthInterval const& interval,
  DateTemplate<TRAITS> date)
{
  return date + interval;
}


template<class TRAITS>
inline DateTemplate<TRAITS>
operator-(
  DateTemplate<TRAITS> date,
  MonthInterval const& interval)
{
  return DateTemplate<TRAITS>::from_datenum((-interval).shift(date.get_datenum()));
}


MonthInterval constexpr MONTH   = MonthInterval(1);
MonthInterval constexpr YEAR    = MonthInterval(12);


//------------------------------------------------------------------------------

}  // namespace cron
}  // namespace alxs

