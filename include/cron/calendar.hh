#pragma once

#include <array>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "cron/date.hh"
#include "filename.hh"

namespace alxs {
namespace cron {

//------------------------------------------------------------------------------
// Declarations

class Calendar;
class HolidayCalendar;

extern HolidayCalendar parse_holiday_calendar(std::istream& in);
extern HolidayCalendar load_holiday_calendar(fs::Filename const& filename);

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
  virtual ~Calendar() {}

  virtual inline Date 
  shift(
    Date date, 
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

  virtual inline Date 
  nearest(
    Date date, 
    bool forward=true) 
    const
  {
    while (date.is_valid() && !contains_(date)) 
      date += forward ? 1 : -1;
    return date;
  }

  template<class DATE> bool contains(DATE date) const { return contains_(Date(date)); }
  template<class DATE> DATE shift(DATE date, ssize_t shift) const { return DATE(this->shift(Date(date), shift)); }
  template<class DATE> DATE nearest(DATE date, bool forward=true) const { return DATE(nearest(Date(date), forward)); }

  template<class DATE> bool operator[](DATE date) const { return contains<DATE>(date); }
  
  CalendarInterval const DAY;

protected:

  virtual bool contains_(Date date) const = 0;

};


template<>
inline bool 
Calendar::contains<Date>(
  Date date) 
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

  virtual Date shift(Date date, ssize_t days) const { return shift(date, days); }

protected:

  virtual bool contains_(Date date) const { return date.is_valid(); }

};


//------------------------------------------------------------------------------

class WeekdaysCalendar
  : public Calendar
{
public:

  typedef std::array<bool, 7> Mask;

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
    Date date) 
    const
  {
    return mask_[date.get_weekday()];
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
    Date min, 
    Date max)
    : min_(min),
      holidays_(max - min, false)
  {
    assert(min.is_valid() && max.is_valid());
  }

  ~HolidayCalendar() {}

  Date get_min() const { return min_; }
  Date get_max() const { return min_ + holidays_.size(); }

  Date 
  shift(
    Date date, 
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
    Date date,
    bool contained)
  {
    ssize_t const index = date - min_;
    if (!(0 <= index && index < (ssize_t) holidays_.size()))
      throw ValueError("date out of calendar range");
    holidays_[index] = contained;
  }

  void add(Date date)       { set(date, true); }
  void remove(Date date)    { set(date, false); }

protected:

  inline bool
  contains_(
    Date date)
    const
  {
    return holidays_[date - min_];
  }


private:

  Date min_;
  std::vector<bool> holidays_;

};


//------------------------------------------------------------------------------
// Class WorkdayCalendar.

class WorkdayCalendar
  : public Calendar
{
public:

  WorkdayCalendar(
    WeekdaysCalendar const& workdays, 
    HolidayCalendar const& holidays)
    : workdays_(workdays),
      holidays_(holidays)
  {
  }

  WorkdayCalendar(
    std::vector<Weekday> const& weekdays,
    fs::Filename const& holidays)
    : WorkdayCalendar(weekdays, load_holiday_calendar(holidays))
  {
  }

  virtual ~WorkdayCalendar() {}
  
protected:

  virtual inline bool 
  contains_(
    Date date) 
    const
  {
    return workdays_.contains(date) && ! holidays_.contains(date);
  }

private:

  WeekdaysCalendar workdays_;
  HolidayCalendar holidays_;

};



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

}  // namespace alxs
}  // namespace cron

