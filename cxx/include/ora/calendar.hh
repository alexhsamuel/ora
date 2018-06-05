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

class Calendar;
class HolidayCalendar;

extern std::unique_ptr<HolidayCalendar> parse_holiday_calendar(std::istream& in);
extern std::unique_ptr<HolidayCalendar> load_holiday_calendar(fs::Filename const& filename);

//------------------------------------------------------------------------------
// Helpers

namespace {

std::pair<Date, Date>
common_range(
  std::pair<Date, Date> range0,
  std::pair<Date, Date> range1)
{
  return {
    std::max(range0.first, range1.first),
    std::min(range0.second, range1.second)
  };
}


}  // anonymous namespace

//------------------------------------------------------------------------------

class Calendar
{
public:

  virtual std::unique_ptr<Calendar> clone() const = 0;

  virtual std::pair<Date, Date> range() const = 0;

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

  virtual bool contains(Date date) const = 0;
  virtual Date before(Date date) const = 0;
  virtual Date after(Date date) const = 0;
  virtual Date shift(Date date, ssize_t shift) const = 0;

};


//------------------------------------------------------------------------------

/*
 * Calendar base that implements before(), after(), and shift() using
 * contains().  This may not be efficient for sparse calendars.
 */
class SimpleCalendar
: public Calendar
{
public:

  virtual inline Date before(
    Date date)
    const override
  {
    while (date.is_valid() && !contains(date))
      date--;
    return date;
  }

  virtual inline Date after(
    Date date)
    const override
  {
    while (date.is_valid() && !contains(date))
      date++;
    return date;
  }

  virtual inline Date 
  shift(
    Date date, 
    ssize_t shift) 
    const override
  {
    while (shift > 0 && date.is_valid())
      if (contains(++date))
        shift--;
    while (shift < 0 && date.is_valid())
      if (contains(--date))
        shift++;
    return date;
  }

};


//------------------------------------------------------------------------------

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

class AllCalendar 
  final
: public Calendar
{
public:

  AllCalendar() {}
  AllCalendar(AllCalendar const&)       = default;
  AllCalendar(AllCalendar&&)            = default;
  virtual ~AllCalendar()                = default;

  virtual std::unique_ptr<Calendar> clone() const override
    { return std::make_unique<AllCalendar>(*this); }
  virtual std::pair<Date, Date> range() const override
    { return {Date::MIN, Date::MAX}; }
  virtual bool contains(Date const date) const override
    { return date.is_valid(); }
  virtual Date before(Date const date) const override
    { return date.is_valid() ? date : Date::INVALID; }
  virtual Date after(Date const date) const override
    { return date.is_valid() ? date : Date::INVALID; }
  virtual Date shift(Date const date, ssize_t const days) const override 
    { return date + days; }

};


//------------------------------------------------------------------------------

class WeekdaysCalendar
: public SimpleCalendar
{
public:

  // FIXME: This can be optimized by storing the before/after shifts from
  // each weekday to the adjacent weekdays.

  WeekdaysCalendar(
    std::vector<Weekday> weekdays)
  {
    mask_.fill(false);
    for (auto const weekday : weekdays)
      mask_[weekday] = true;
  }

  virtual ~WeekdaysCalendar() {}

  virtual std::unique_ptr<Calendar> clone() const override
    { return std::make_unique<WeekdaysCalendar>(*this); }
  virtual std::pair<Date, Date> range() const override
    { return {Date::MIN, Date::MAX}; }
  virtual bool contains(Date const date) const override
    { return mask_[get_weekday(date)]; }

private:

  std::array<bool, 7> mask_;

};


//------------------------------------------------------------------------------

class HolidayCalendar
  final 
: public SimpleCalendar
{
public:

  HolidayCalendar(
    date::Date const min, 
    date::Date const max)
  : min_(min),
    holidays_(max - min, false)
  {
    assert(min.is_valid() && max.is_valid());
  }

  ~HolidayCalendar() {}

  virtual std::unique_ptr<Calendar> clone() const override
    { return std::make_unique<HolidayCalendar>(*this); }

  virtual std::pair<Date, Date> 
  range() 
    const override
  {
    return {min_, min_ + holidays_.size()};
  }

  inline virtual bool
  contains(
    Date date)
    const override
  {
    return holidays_[date - min_];
  }

  date::Date 
  shift(
    date::Date date, 
    ssize_t shift) 
    const override
  {
    while (shift > 0 && date.is_valid())
      if (contains(++date))
        shift--;
    while (shift < 0 && date.is_valid())
      if (contains(--date))
        shift++;
    return date;
  }

  // Mutators

  inline void
  set(
    date::Date const date,
    bool const contained)
  {
    ssize_t const index = date - min_;
    if (!(0 <= index && index < (ssize_t) holidays_.size()))
      throw ValueError("date out of calendar range");
    holidays_[index] = contained;
  }

  void add(Date date)       { set(date, true); }
  void remove(Date date)    { set(date, false); }

private:

  Date min_;
  std::vector<bool> holidays_;

};


template<class LineIter>
std::unique_ptr<HolidayCalendar>
parse_holiday_calendar(
  LineIter&& lines,
  LineIter&& end)
{
  std::vector<Date> dates;
  Date min = Date::MISSING;
  Date max = Date::MISSING;
  Date date_min = Date::MISSING;
  Date date_max = Date::MISSING;

  for (; lines != end; ++lines) {
    auto line = strip(*lines);
    // Skip blank and comment lines.
    if (line.size() == 0 || line[0] == '#')
      continue;
    StringPair parts = split1(line);
    // FIXME: Handle exceptions.
    if (parts.first == "MIN") 
      min = date::from_iso_date<Date>(parts.second);
    else if (parts.first == "MAX")
      max = date::from_iso_date<Date>(parts.second);
    else {
      auto const date = date::from_iso_date<Date>(parts.first);
      dates.push_back(date);
      // Keep track of the min and max dates we've seen.
      if (!date::nex::before(date_min, date))
        date_min = date;
      if (!date::nex::before(date, date_max))
        date_max = date + 1;
    }
  }

  // Infer missing min or max from the range of given dates.
  if (min.is_missing()) 
    min = dates.size() > 0 ? date_min : Date::MIN;
  assert(!min.is_missing());
  if (max.is_missing()) 
    max = dates.size() > 0 ? date_max : Date::MIN;
  assert(!max.is_missing());

  // Now construct the calendar.
  assert(min <= max);
  auto cal = std::make_unique<HolidayCalendar>(min, max);
  for (auto const date : dates)
    cal->add(date);
  return cal;
}


//------------------------------------------------------------------------------

class NegationCalendar
: public SimpleCalendar
{
public:
  
  NegationCalendar(
    std::unique_ptr<Calendar>&& calendar)
  : calendar_(std::move(calendar))
  {
  }
  
  NegationCalendar(
    NegationCalendar const& calendar)
  : calendar_(calendar.calendar_->clone())
  {
  }

  NegationCalendar& operator=(NegationCalendar const&) = delete;
  NegationCalendar& operator=(NegationCalendar&&) = delete;
  virtual ~NegationCalendar() = default;

  virtual std::unique_ptr<Calendar> clone() const override
    { return std::make_unique<NegationCalendar>(*this); }
  virtual inline std::pair<Date, Date> range() const override
    { return calendar_->range(); }
  virtual inline bool contains(Date const date) const override
    { return !calendar_->contains(date); }

private:

  std::unique_ptr<Calendar> const calendar_;

};


class UnionCalendar
: public SimpleCalendar
{
public:

  UnionCalendar(
    std::unique_ptr<Calendar>&& calendar0,
    std::unique_ptr<Calendar>&& calendar1)
  : cal0_(std::move(calendar0)),
    cal1_(std::move(calendar1))
  {
  }

  UnionCalendar(
    UnionCalendar const& calendar)
  : cal0_(calendar.cal0_->clone())
  , cal1_(calendar.cal1_->clone())
  {
  }

  UnionCalendar& operator=(UnionCalendar const&) = delete;
  UnionCalendar& operator=(UnionCalendar&&) = delete;
  virtual ~UnionCalendar() = default;

  virtual std::unique_ptr<Calendar> clone() const override
    { return std::make_unique<UnionCalendar>(*this); }
  virtual inline std::pair<Date, Date> range() const override
    { return common_range(cal0_->range(), cal1_->range()); }
  virtual inline bool contains(Date const date) const override
    { return cal0_->contains(date) || cal1_->contains(date); }

private:

  std::unique_ptr<Calendar> cal0_;
  std::unique_ptr<Calendar> cal1_;

};


class IntersectionCalendar
: public SimpleCalendar
{
public:

  IntersectionCalendar(
    std::unique_ptr<Calendar>&& calendar0,
    std::unique_ptr<Calendar>&& calendar1)
  : cal0_(std::move(calendar0)),
    cal1_(std::move(calendar1))
  {
  }

  IntersectionCalendar(
    IntersectionCalendar const& calendar)
  : cal0_(calendar.cal0_->clone())
  , cal1_(calendar.cal1_->clone())
  {
  }

  IntersectionCalendar& operator=(IntersectionCalendar const&) = delete;
  IntersectionCalendar& operator=(IntersectionCalendar&&) = delete;
  virtual ~IntersectionCalendar() = default;

  virtual std::unique_ptr<Calendar> clone() const override
    { return std::make_unique<IntersectionCalendar>(*this); }
  virtual inline std::pair<Date, Date> range() const override
    { return common_range(cal0_->range(), cal1_->range()); }
  virtual inline bool contains(Date const date) const override
    { return cal0_->contains(date) && cal1_->contains(date); }

private:

  std::unique_ptr<Calendar> cal0_;
  std::unique_ptr<Calendar> cal1_;

};


/*
 * Creates a working calendar, including workdays but with holidays removed.
 *
 * Returns a new calendar which contains all weekdays specified by `weekdays`
 * but with all days in `holidays` removed.
 */
inline std::unique_ptr<Calendar>
make_workday_calendar(
  std::vector<Weekday> weekdays,
  std::unique_ptr<Calendar>&& holidays)
{
  return std::make_unique<IntersectionCalendar>(
    std::make_unique<WeekdaysCalendar>(weekdays),
    std::make_unique<NegationCalendar>(std::move(holidays)));
}


//------------------------------------------------------------------------------
// Functions.

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

//------------------------------------------------------------------------------

inline constexpr Calendar::Interval 
operator-(
  Calendar::Interval const& interval) 
{ 
  return Calendar::Interval(interval.get_calendar(), -interval.get_days()); 
}


inline constexpr Calendar::Interval 
operator*(
  Calendar::Interval const& interval,
  ssize_t mult) 
{ 
  return Calendar::Interval(interval.get_calendar(), mult * interval.get_days()); 
}


inline constexpr Calendar::Interval
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

