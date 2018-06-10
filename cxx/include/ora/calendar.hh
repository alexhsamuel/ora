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
#include "ora/lib/filename.hh"
#include "ora/lib/string.hh"

namespace ora {

using namespace ora::lib;
using ora::date::Date;

//------------------------------------------------------------------------------
// Declarations

template<class T> struct Range;
class Calendar;

extern Calendar load_calendar(fs::Filename const& filename);
extern Calendar make_const_calendar(Range<Date>, bool);
extern Calendar make_weekday_calendar(Range<Date>, bool const[7]);

//------------------------------------------------------------------------------
// Helpers

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
// Calendar file

/*
  Holiday calendar file format:
    - Line-oriented text, delimited by NL.
    - Leading and trailing whitespace on each line stripped.
    - Blank lines ignored.
    - Lines beginning with # ignored as comment lines.
    - All dates specified as ISO dates, 'YYYY-MM-DD'
    - Range optionally specified with lines 'MIN <date>' and 'MAX <date>'.
    - Every other line consists of a holiday date followed by whitespace;
      the rest of the line is ignored.
    - If range min or max is not specified, it is inferred from the dates.

  Example:

      # U.S. holidays for the year 2010.

      MIN 2010-01-01
      MAX 2011-01-01

      2010-01-01 New Year's Day
      2010-01-18 Birthday of Martin Luther King, Jr.
      2010-02-15 Washington's Birthday
      2010-05-31 Memorial Day
      2010-07-05 Independence Day
      2010-09-06 Labor Day
      2010-10-11 Columbus Day
      2010-11-11 Veterans Day
      2010-11-25 Thanksgiving Day
      2010-12-24 Christmas Day
      2010-12-31 New Year's Day
*/

template<class LineIter>
Calendar
parse_calendar(
  LineIter&& lines,
  LineIter&& end)
{
  std::vector<Date> dates;
  auto range = Range<Date>{Date::MISSING, Date::MISSING};
  auto date_range = Range<Date>{Date::MISSING, Date::MISSING};

  for (; lines != end; ++lines) {
    auto line = strip(*lines);
    // Skip blank and comment lines.
    if (line.size() == 0 || line[0] == '#')
      continue;
    StringPair parts = split1(line);
    // FIXME: Handle exceptions.
    if (parts.first == "MIN") 
      range.min = date::from_iso_date<Date>(parts.second);
    else if (parts.first == "MAX")
      range.max = date::from_iso_date<Date>(parts.second);
    else {
      auto const date = date::from_iso_date<Date>(parts.first);
      dates.push_back(date);
      // Keep track of the min and max dates we've seen.
      if (!date::nex::before(date_range.min, date))
        date_range.min = date;
      if (!date::nex::before(date, date_range.max))
        date_range.max = date + 1;
    }
  }

  // Infer missing min or max from the range of given dates.
  if (range.min.is_missing()) 
    range.min = dates.size() > 0 ? date_range.min : Date::MIN;
  if (range.max.is_missing()) 
    range.max = dates.size() > 0 ? date_range.max : Date::MIN;
  // FIXME: Exceptions instead.
  assert(!range.min.is_missing());
  assert(!range.max.is_missing());
  assert(range.min <= range.max);

  // Now construct the calendar.
  return {range, dates};
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

