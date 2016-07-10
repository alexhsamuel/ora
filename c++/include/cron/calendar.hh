#pragma once

#include <array>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "aslib/filename.hh"
#include "cron/date_functions.hh"
#include "cron/date_type.hh"

namespace cron {

using namespace aslib;

//------------------------------------------------------------------------------
// Declarations

class Calendar;
class HolidayCalendar;

extern std::unique_ptr<HolidayCalendar> parse_holiday_calendar(std::istream& in);
extern std::unique_ptr<HolidayCalendar> load_holiday_calendar(fs::Filename const& filename);

//------------------------------------------------------------------------------

class CalendarInterval
{
public:

  constexpr 
  CalendarInterval(
    Calendar const& calendar, 
    ssize_t days) 
    : calendar_(calendar), 
      days_(days) 
  {
  }

  constexpr Calendar const& get_calendar() const { return calendar_; }
  constexpr ssize_t get_days() const { return days_; }

private:

  Calendar const& calendar_;
  ssize_t days_;

};


//------------------------------------------------------------------------------

class Calendar
{
public:

  Calendar() : DAY(*this, 1) {}
  Calendar(Calendar const&) = delete;
  Calendar(Calendar&&) = delete;
  Calendar& operator=(Calendar const&) = delete;
  Calendar& operator=(Calendar&&) = delete;
  virtual ~Calendar() {}

  virtual inline date::Date 
  shift(
    date::Date date, 
    ssize_t shift) 
    const
  {
    // FIXME: What if 'date' is not in the calendar?
    // FIXME: Avoid virtual calls to contains()?

    while (shift > 0 && date.is_valid())
      if (contains_(++date))
        shift--;
    while (shift < 0 && date.is_valid())
      if (contains_(--date))
        shift++;
    return date;
  }

  virtual inline date::Date 
  nearest(
    date::Date date, 
    bool forward=true) 
    const
  {
    while (date.is_valid() && !contains_(date)) 
      date += forward ? 1 : -1;
    return date;
  }

  template<class DATE> bool contains(DATE date) const { return contains_(date::Date(date)); }
  template<class DATE> DATE shift(DATE date, ssize_t shift) const { return DATE(this->shift(date::Date(date), shift)); }
  template<class DATE> DATE nearest(DATE date, bool forward=true) const { return DATE(nearest(date::Date(date), forward)); }

  template<class DATE> bool operator[](DATE date) const { return contains<DATE>(date); }
  
  CalendarInterval const DAY;

protected:

  virtual bool contains_(date::Date date) const = 0;

};


template<>
inline bool 
Calendar::contains<date::Date>(
  date::Date date) 
  const 
{ 
  return contains_(date); 
}


template<class DATE>
inline DATE
operator<<(
  DATE date,
  Calendar const& cal)
{
  return cal.nearest(date, false);
}


template<class DATE>
inline DATE
operator>>(
  DATE date,
  Calendar const& cal)
{
  return cal.nearest(date, true);
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
  virtual ~AllCalendar() {}

  virtual date::Date shift(date::Date date, ssize_t days) const { return shift(date, days); }

protected:

  virtual bool contains_(date::Date date) const { return date.is_valid(); }

};


//------------------------------------------------------------------------------

class WeekdaysCalendar
  : public Calendar
{
public:

  using Mask = std::array<bool, 7>;

  WeekdaysCalendar(
    std::vector<Weekday> weekdays)
  {
    mask_.fill(false);
    for (auto const weekday : weekdays)
      mask_[weekday] = true;
  }

  virtual ~WeekdaysCalendar() {}

  // FIXME: Optimize shift()?

protected:

  virtual inline bool 
  contains_(
    date::Date date) 
    const
  {
    return mask_[get_weekday(date)];
  }

private:

  Mask mask_;

};


//------------------------------------------------------------------------------

class HolidayCalendar
  final 
  : public Calendar
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

  date::Date get_min() const { return min_; }
  date::Date get_max() const { return min_ + holidays_.size(); }

  date::Date 
  shift(
    date::Date date, 
    ssize_t shift) 
    const
  {
    while (shift > 0 && date.is_valid())
      if (contains_(++date))
        shift--;
    while (shift < 0 && date.is_valid())
      if (contains_(--date))
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

  void add(date::Date date)       { set(date, true); }
  void remove(date::Date date)    { set(date, false); }

protected:

  inline bool
  contains_(
    date::Date date)
    const
  {
    return holidays_[date - min_];
  }


private:

  date::Date min_;
  std::vector<bool> holidays_;

};


//------------------------------------------------------------------------------

class NegationCalendar
  : public Calendar
{
public:
  
  NegationCalendar(
    std::unique_ptr<Calendar>&& calendar)
  : calendar_(std::move(calendar))
  {
  }
  
  NegationCalendar& operator=(NegationCalendar const&) = delete;
  NegationCalendar& operator=(NegationCalendar&&) = delete;
  virtual ~NegationCalendar() = default;

protected:

  virtual inline bool
  contains_(
    date::Date const date)
    const
  {
    return !calendar_->contains(date);
  }

private:

  std::unique_ptr<Calendar> const calendar_;

};


class UnionCalendar
  : public Calendar
{
public:

  UnionCalendar(
    std::unique_ptr<Calendar>&& calendar0,
    std::unique_ptr<Calendar>&& calendar1)
  : calendar0_(std::move(calendar0)),
    calendar1_(std::move(calendar1))
  {
  }

  UnionCalendar& operator=(UnionCalendar const&) = delete;
  UnionCalendar& operator=(UnionCalendar&&) = delete;
  virtual ~UnionCalendar() = default;

protected:

  virtual inline bool
  contains_(
    date::Date const date)
    const
  {
    return calendar0_->contains(date) && calendar1_->contains(date);
  }

private:

  std::unique_ptr<Calendar> calendar0_;
  std::unique_ptr<Calendar> calendar1_;

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
  return std::make_unique<UnionCalendar>(
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

inline constexpr CalendarInterval 
operator-(
  CalendarInterval const& interval) 
{ 
  return CalendarInterval(interval.get_calendar(), -interval.get_days()); 
}


inline constexpr CalendarInterval 
operator*(
  CalendarInterval const& interval,
  ssize_t mult) 
{ 
  return CalendarInterval(interval.get_calendar(), mult * interval.get_days()); 
}


inline constexpr CalendarInterval
operator*(
  ssize_t mult,
  CalendarInterval const& interval)
{
  return interval * mult;
}


template<class DATE>
inline DATE
operator+(
  DATE date,
  CalendarInterval const& interval)
{
  return interval.get_calendar().shift(date, interval.get_days());
}


template<class DATE>
inline DATE
operator+(
  CalendarInterval const& interval,
  DATE date)
{
  return date + interval;
}


template<class DATE>
inline DATE
operator-(
  DATE date,
  CalendarInterval const& interval)
{
  return date + -interval;
}


//------------------------------------------------------------------------------

}  // namespace cron

