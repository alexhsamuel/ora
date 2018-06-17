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

template<class T> struct Interval;
class Calendar;

extern Calendar parse_calendar(ora::lib::Iter<std::string>&);
extern Calendar load_calendar(fs::Filename const& filename);
extern Calendar make_const_calendar(Interval<Date>, bool);
extern Calendar make_weekday_calendar(Interval<Date>, bool const[7]);

//------------------------------------------------------------------------------
// Helpers

/*
 * Inclusive (min, max) interval.
 */
template<class T>
struct Interval
{
  Interval(
    T const start_, 
    T const stop_) 
  : start(start_)
  , stop(stop_) 
  { 
  }

  int length() const { return stop - start; }
  bool contains(T const val) const { return start <= val && val < stop; }

  T start;
  T stop;

};


template<class T>
inline Interval<T>
operator&(
  Interval<T> interval0,
  Interval<T> interval1)
{
  auto const start = std::min(interval0.start, interval1.start);
  auto const stop = std::max(interval0.stop, interval1.stop);
  return {start, std::max(start, stop)};
}


//------------------------------------------------------------------------------

class Calendar
{
public:

  Calendar(
    date::Date const start,
    std::vector<bool>&& dates)
  : start_(start)
  , dates_(std::move(dates))
  {
  }

  Calendar(
    Interval<Date> range,
    std::vector<Date> const& dates)
  : start_(range.start)
  , dates_(range.stop - range.start, false)
  {
    assert(range.start.is_valid() && range.stop.is_valid());
    for (auto date : dates) {
      if (!range.contains(date))
        throw ValueError("date out of calendar range");
      dates_[date - start_] = true;
    }
  }

  Calendar(Calendar const&)                 = default;
  Calendar(Calendar&&)                      = default;
  ~Calendar()                               = default;

  Interval<Date>
  range() 
    const
  {
    return {start_, start_ + dates_.size()};
  }

  bool
  contains(
    Date date)
    const
  {
    if (   !date.is_valid() 
        || date < start_ 
        || (size_t) (date - start_) >= dates_.size())
      throw CalendarRangeError();
    else
      return dates_[date - start_];
  }

  Date 
  before(
    Date date)
    const
  {
    while (!contains(date))
      date--;
    return date;
  }

  Date 
  after(
    Date date)
    const
  {
    while (!contains(date))
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

  class Day
  {
  public:

    constexpr Day(Calendar const& calendar, ssize_t const days) 
      : calendar_(calendar), days_(days) {}

    Day(Day const&) = default;
    Day& operator=(Day&) = default;

    constexpr Calendar const& get_calendar() const { return calendar_; }
    constexpr ssize_t get_days() const { return days_; }

  private:

    Calendar const& calendar_;
    ssize_t days_;

  };


  Day DAY() const { return Day(*this, 1); }

private:

  friend Calendar operator!(Calendar const&);
  friend Calendar operator&(Calendar const&, Calendar const&);
  friend Calendar operator|(Calendar const&, Calendar const&);
  friend Calendar operator^(Calendar const&, Calendar const&);

  Date start_;
  std::vector<bool> dates_;

};


inline Calendar
operator!(
  Calendar const& cal)
{
  auto dates = cal.dates_;
  dates.flip();
  return {cal.start_, std::move(dates)};
}


inline Calendar
operator&(
  Calendar const& cal0,
  Calendar const& cal1)
{
  auto const range  = cal0.range() & cal1.range();
  auto const length = range.length();
  auto const off0   = range.start - cal0.start_;
  auto const off1   = range.start - cal1.start_;

  auto dates = std::vector<bool>(length);
  for (auto i = 0; i < length; ++i)
    dates[i] = cal0.dates_[off0 + i] && cal1.dates_[off1 + i];

  return {range.start, std::move(dates)};
}


inline Calendar
operator|(
  Calendar const& cal0,
  Calendar const& cal1)
{
  auto const range  = cal0.range() & cal1.range();
  auto const length = range.length();
  auto const off0   = range.start - cal0.start_;
  auto const off1   = range.start - cal1.start_;

  auto dates = std::vector<bool>(length);
  for (auto i = 0; i < length; ++i)
    dates[i] = cal0.dates_[off0 + i] || cal1.dates_[off1 + i];

  return {range.start, std::move(dates)};
}


inline Calendar
operator^(
  Calendar const& cal0,
  Calendar const& cal1)
{
  auto const range  = cal0.range() & cal1.range();
  auto const length = range.length();
  auto const off0   = range.start - cal0.start_;
  auto const off1   = range.start - cal1.start_;

  auto dates = std::vector<bool>(length);
  for (auto i = 0; i < length; ++i)
    dates[i] = cal0.dates_[off0 + i] ^ cal1.dates_[off1 + i];

  return {range.start, std::move(dates)};
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
Calendar::Day 
operator-(
  Calendar::Day const& interval) 
{ 
  return Calendar::Day(interval.get_calendar(), -interval.get_days()); 
}


inline constexpr
Calendar::Day 
operator*(
  Calendar::Day const& interval,
  ssize_t mult) 
{ 
  return Calendar::Day(interval.get_calendar(), mult * interval.get_days()); 
}


inline constexpr
Calendar::Day
operator*(
  ssize_t mult,
  Calendar::Day const& interval)
{
  return interval * mult;
}


template<class DATE>
inline DATE
operator+(
  DATE date,
  Calendar::Day const& interval)
{
  return interval.get_calendar().shift(date, interval.get_days());
}


template<class DATE>
inline DATE
operator+(
  Calendar::Day const& interval,
  DATE date)
{
  return date + interval;
}


template<class DATE>
inline DATE
operator-(
  DATE date,
  Calendar::Day const& interval)
{
  return date + -interval;
}


//------------------------------------------------------------------------------

}  // namespace ora

