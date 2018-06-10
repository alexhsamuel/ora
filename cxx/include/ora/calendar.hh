#pragma once

#include <array>
#include <iostream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

#include "ora/date_functions.hh"
#include "ora/date_type.hh"
#include "ora/date_nex.hh"
#include "ora/lib/file.hh"
#include "ora/lib/filename.hh"
#include "ora/lib/string.hh"

namespace ora {

using namespace ora::lib;
using ora::date::Date;

//------------------------------------------------------------------------------
// Declarations

template<class T> struct Range;
class Calendar;

extern Calendar parse_calendar(ora::lib::Iter<std::string>&);
extern Calendar load_calendar(fs::Filename const& filename);
extern Calendar make_const_calendar(Range<Date>, bool);
extern Calendar make_weekday_calendar(Range<Date>, bool const[7]);

//------------------------------------------------------------------------------
// Helpers

/*
 * Inclusive (min, max) range.
 */
template<class T>
struct Range
{
  Range(
    T const mn, 
    T const mx) 
  : min(mn)
  , max(mx) 
  { 
    assert(min <= max); 
  }

  bool contains(T const val) const { return min <= val && val <= max; }

  T min;
  T max;

};


template<class T>
inline Range<T>
operator&(
  Range<T> range0,
  Range<T> range1)
{
  auto const min = std::min(range0.min, range1.min);
  auto const max = std::max(range0.max, range1.max);
  return {min, std::max(min, max)};
}


//------------------------------------------------------------------------------

class Calendar
{
public:

  Calendar(
    date::Date const min,
    std::vector<bool>&& dates)
  : min_(min)
  , dates_(std::move(dates))
  {
  }

  Calendar(
    Range<Date> range,
    std::vector<Date> const& dates)
  : min_(range.min)
  , dates_(range.max - range.min, false)
  {
    assert(range.min.is_valid() && range.max.is_valid());
    for (auto date : dates) {
      if (date < range.min || range.max < date)
        throw ValueError("date out of calendar range");
      dates_[date - min_] = true;
    }
  }

  Calendar(Calendar const&)                 = default;
  Calendar(Calendar&&)                      = default;
  ~Calendar()                               = default;

  Range<Date>
  range() 
    const
  {
    return {min_, min_ + dates_.size()};
  }

  bool
  contains(
    Date date)
    const
  {
    if (!date.is_valid())
      return false;
    else if (date < min_ || date - min_ >= dates_.size())
      throw CalendarRangeError();
    else
      return dates_[date - min_];
  }

  Date 
  before(
    Date date)
    const
  {
    while (date.is_valid() && !contains(date))
      date--;
    return date;
  }

  Date 
  after(
    Date date)
    const
  {
    while (date.is_valid() && !contains(date))
      date++;
    return date;
  }

  date::Date 
  shift(
    date::Date date, 
    ssize_t shift) 
    const
  {
    while (shift > 0 && date.is_valid())
      if (contains(++date))
        shift--;
    while (shift < 0 && date.is_valid())
      if (contains(--date))
        shift++;
    return date;
  }

  // FIXME: Check range.
  template<class DATE> bool contains(DATE date) const 
    { return contains(Date(date)); }
  template<class DATE> DATE before(DATE const date) const 
    { return date.is_valid() ? DATE(before(Date(date))) : DATE::INVALID; }
  template<class DATE> DATE after(DATE const date) const
    { return date.is_valid() ? DATE(after(Date(date))) : DATE::INVALID; }
  template<class DATE> DATE shift(DATE const date, ssize_t const count) const 
    { return date.is_valid() ? DATE(shift(Date(date), count)) : DATE::INVALID; }

  class Interval
  {
  public:

    constexpr Interval(Calendar const& calendar, ssize_t const days) 
      : calendar_(calendar), days_(days) {}

    Interval(Interval const&) = default;
    Interval& operator=(Interval&) = default;

    constexpr Calendar const& get_calendar() const { return calendar_; }
    constexpr ssize_t get_days() const { return days_; }

  private:

    Calendar const& calendar_;
    ssize_t days_;

  };


  Interval DAY() const { return Interval(*this, 1); }

private:

  friend Calendar operator!(Calendar const&);
  friend Calendar operator&(Calendar const&, Calendar const&);
  friend Calendar operator|(Calendar const&, Calendar const&);

  Date min_;
  std::vector<bool> dates_;

};


inline Calendar
operator!(
  Calendar const& cal)
{
  auto dates = cal.dates_;
  dates.flip();
  return {cal.min_, std::move(dates)};
}


inline Calendar
operator&(
  Calendar const& cal0,
  Calendar const& cal1)
{
  auto const range  = cal0.range() & cal1.range();
  auto const length = range.max - range.min;
  auto const off0   = range.min - cal0.min_;
  auto const off1   = range.min - cal1.min_;

  auto dates = std::vector<bool>(length);
  for (auto i = 0; i < length; ++i)
    dates[i] = cal0.dates_[off0 + i] && cal1.dates_[off1 + i];

  return {range.min, std::move(dates)};
}


inline Calendar
operator|(
  Calendar const& cal0,
  Calendar const& cal1)
{
  auto const range  = cal0.range() & cal1.range();
  auto const length = range.max - range.min;
  auto const off0   = range.min - cal0.min_;
  auto const off1   = range.min - cal1.min_;

  auto dates = std::vector<bool>(length);
  for (auto i = 0; i < length; ++i)
    dates[i] = cal0.dates_[off0 + i] || cal1.dates_[off1 + i];

  return {range.min, std::move(dates)};
}


//------------------------------------------------------------------------------
// Functions.

template<class DATE>
inline DATE
operator<<(
  DATE date,
  Calendar const& cal)
{
  return cal.before(date - 1);
}


template<class DATE>
inline DATE
operator>>(
  DATE date,
  Calendar const& cal)
{
  return cal.after(date + 1);
}


template<class DATE>
inline DATE
operator<<=(
  DATE& date,
  Calendar const& cal)
{
  return date = date << cal;
}


template<class DATE>
inline DATE
operator>>=(
  DATE& date,
  Calendar const& cal)
{
  return date = date >> cal;
}


//------------------------------------------------------------------------------
// Interval functions

inline constexpr
Calendar::Interval 
operator-(
  Calendar::Interval const& interval) 
{ 
  return Calendar::Interval(interval.get_calendar(), -interval.get_days()); 
}


inline constexpr
Calendar::Interval 
operator*(
  Calendar::Interval const& interval,
  ssize_t mult) 
{ 
  return Calendar::Interval(interval.get_calendar(), mult * interval.get_days()); 
}


inline constexpr
Calendar::Interval
operator*(
  ssize_t mult,
  Calendar::Interval const& interval)
{
  return interval * mult;
}


template<class DATE>
inline DATE
operator+(
  DATE date,
  Calendar::Interval const& interval)
{
  return interval.get_calendar().shift(date, interval.get_days());
}


template<class DATE>
inline DATE
operator+(
  Calendar::Interval const& interval,
  DATE date)
{
  return date + interval;
}


template<class DATE>
inline DATE
operator-(
  DATE date,
  Calendar::Interval const& interval)
{
  return date + -interval;
}


//------------------------------------------------------------------------------

}  // namespace ora

