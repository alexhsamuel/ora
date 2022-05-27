#pragma once

#include "ora/date_math.hh"
#include "ora/date_type.hh"
#include "ora/types.hh"

namespace ora {
namespace date {

//------------------------------------------------------------------------------
// Factory functions
//------------------------------------------------------------------------------

// Forward declarations.
template<class DATE=Date> DATE from_ordinal_date(Year, Ordinal);
template<class DATE=Date> DATE from_ymd(Year, Month, Day);
template<class DATE=Date> DATE from_week_date(Year, Week, Weekday);

template<class DATE=Date> inline DATE from_ordinal_date(OrdinalDate const& d)
  { return from_ordinal_date<DATE>(d.year, d.ordinal); }
template<class DATE=Date> inline DATE from_ymd(YmdDate const& d)
  { return from_ymd<DATE>(d.year, d.month, d.day); }
template<class DATE=Date> inline DATE from_week_date(WeekDate const& d)
  { return from_week_date<DATE>(d.week_year, d.week, d.weekday); }

// Synonyms for static factory methods; included for completeness.
template<class DATE=Date> inline DATE from_datenum(Datenum const d)
  { return DATE::from_datenum(d); }
template<class DATE=Date> inline DATE from_offset(typename DATE::Offset const o)
  { return DATE::from_offset(o); }

// FIXME: Remove.
template<class DATE=Date> 
inline DATE 
from_iso_date(
  std::string const& date)
{ 
  auto ymd = parse_iso_date(date);
  if (year_is_valid(ymd.year))
    return from_ymd<DATE>(ymd);
  else
    throw DateFormatError("not ISO date format");
}


/*
 * Creates a date from an ordinal date.
 *
 * Throws <InvalidDateError> if the ordinal date is invalid.
 * Throws <DateRangeError> if the ordinal date is out of range.
 */
template<class DATE> 
inline DATE 
from_ordinal_date(
  Year const year,
  Ordinal const ordinal)
{ 
  if (ordinal_date_is_valid(year, ordinal))
    return from_datenum<DATE>(ordinal_date_to_datenum(year, ordinal));
  else
    throw InvalidDateError();
}


/*
 * Creates a date from a week date.
 *
 * Throws <InvalidDateError> if the week date is invalid.
 * Throws <DateRangeError> if the week date is out of range.
 */
template<class DATE> 
inline DATE 
from_week_date(
  Year const week_year,
  Week const week,
  Weekday const weekday)
{
  if (week_date_is_valid(week_year, week, weekday))
    return from_datenum<DATE>(week_date_to_datenum(week_year, week, weekday));
  else
    throw InvalidDateError();
}


/*
 * Creates a date from year, month, and day.
 *
 * Throws <InvalidDateError> if the year, month, and day are invalid.
 * Throws <DateRangeError> if the date is out of range.
 */
template<class DATE> 
inline DATE 
from_ymd(
  Year const year,
  Month const month,
  Day const day)
{
  if (ymd_is_valid(year, month, day))
    return from_datenum<DATE>(ymd_to_datenum(year, month, day));
  else
    throw InvalidDateError();
}


/*
 * Creates a date from a YMDI.
 *
 * Throws <InvalidDateError> if the YMDI is invalid.
 * Throws <DateRangeError> if the YMDI is out of range.
 */
template<class DATE=Date>
inline DATE
from_ymdi(
  int const ymdi)
{
  if (ymdi_is_valid(ymdi)) 
    return from_datenum<DATE>(ymdi_to_datenum(ymdi));
  else
    throw InvalidDateError();
}


//------------------------------------------------------------------------------
// Accessors
//------------------------------------------------------------------------------

template<class DATE>
inline OrdinalDate 
get_ordinal_date(
  DATE const date)
{ 
  ensure_valid(date);
  return datenum_to_ordinal_date(date.get_datenum());
}


template<class DATE>
inline Weekday 
get_weekday(
  DATE const date)
{ 
  ensure_valid(date);
  return ora::get_weekday(date.get_datenum());
}


template<class DATE>
inline WeekDate 
get_week_date(
  DATE const date)
{ 
  ensure_valid(date);
  return ora::datenum_to_week_date(date.get_datenum());
}


template<class DATE>
inline YmdDate 
get_ymd(
  DATE const date)
{ 
  ensure_valid(date);
  return datenum_to_ymd(date.get_datenum()); 
}


template<class DATE>
inline int 
get_ymdi(
  DATE const date)
{ 
  ensure_valid(date);
  return ora::datenum_to_ymdi(date.get_datenum()); 
}


// For convenience.

template<class DATE> inline Datenum get_datenum(DATE const date)
  { return date.get_datenum(); }
template<class DATE> inline typename DATE::Offset get_offset(DATE const date)
  { return date.get_offset(); }

template<class DATE> inline Day get_day(DATE const date) 
  { return get_ymd(date).day; }
template<class DATE> inline Month get_month(DATE const date) 
  { return get_ymd(date).month; }
template<class DATE> inline Ordinal get_ordinal(DATE const date)
  { return get_ordinal_date(date).ordinal; }
template<class DATE> inline Week get_week(DATE const date)
  { return get_week_date(date).week; }
template<class DATE> inline Year get_week_year(DATE const date)
  { return get_week_date(date).week_year; }
template<class DATE> inline Year get_year(DATE const date) 
  { return get_ordinal_date(date).year; }

//------------------------------------------------------------------------------
// Comparisons
//------------------------------------------------------------------------------

template<class DATE>
inline int
compare(
  DATE const date0,
  DATE const date1)
{
  ensure_valid(date0);
  ensure_valid(date1);
  return compare(date0.get_offset(), date1.get_offset());
}


template<class DATE>
inline bool 
equal(
  DATE const date0,
  DATE const date1)
{ 
  ensure_valid(date0);
  ensure_valid(date1);
  return date0.get_offset() == date1.get_offset();
}


template<class DATE>
inline bool 
before(
  DATE const date0,
  DATE const date1)
{
  ensure_valid(date0);
  ensure_valid(date1);
  return date0.get_offset() < date1.get_offset();
}


//------------------------------------------------------------------------------
// Day arithmetic
//------------------------------------------------------------------------------

template<class DATE>
inline DATE
days_after(
  DATE const date,
  int const days)
{
  ensure_valid(date);
  return from_offset<DATE>(date.get_offset() + days);
}


template<class DATE>
inline DATE
days_before(
  DATE const date,
  int const days)
{
  return days_after(date, -days);
}


template<class DATE>
inline int
days_between(
  DATE const date0,
  DATE const date1)
{
  ensure_valid(date0);
  ensure_valid(date1);
  return (int) date1.get_offset() - date0.get_offset();
}


template<class T>
inline DateTemplate<T>
operator+(DateTemplate<T> const date, int const days)
{
  return days_after(date, days);
}

template<class T>
inline DateTemplate<T>
operator-(DateTemplate<T> const date, int const days)
{
  return days_before(date, days);
}

template<class T>
inline int
operator-(DateTemplate<T> const date1, DateTemplate<T> const date0)
{
  return days_between(date0, date1);
}

template<class T>
inline DateTemplate<T>
operator+=(DateTemplate<T>& date, int const days) 
{
  return date = date + days;
}

template<class T>
inline DateTemplate<T>
operator++(DateTemplate<T>& date) 
{
  return date = date + 1;
}

template<class T>
inline DateTemplate<T>
operator++(DateTemplate<T>& date, int /* tag */) 
{
  auto old = date;
  date = date + 1;
  return old;
}

template<class T>
inline DateTemplate<T>
operator-=(DateTemplate<T>& date, int const days) 
{
  return date = date -days;
}

template<class T>
inline DateTemplate<T>
operator--(DateTemplate<T>& date) 
{
  return date = date - 1;
}

template<class T>
inline DateTemplate<T>
operator--(DateTemplate<T>& date, int /* tag */) 
{
  auto old = date;
  date = date - 1;
  return old;
}

//------------------------------------------------------------------------------

}  // namespace date
}  // namespace ora

