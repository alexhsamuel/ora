#pragma once

#include "cron/date.hh"
#include "cron/date_math.hh"
#include "cron/types.hh"

namespace cron {
namespace date {

//------------------------------------------------------------------------------
// Forward declarations
//------------------------------------------------------------------------------

template<class DATE> DATE from_offset(typename DATE::Offset);
template<class DATE> DATE from_ymd(YmdDate const&);

//------------------------------------------------------------------------------

template<class DATE>
inline DATE
from_datenum(
  Datenum const datenum)
{
  using Offset = typename DATE::Offset;

  if (datenum_is_valid(datenum)) {
    auto offset = (long) datenum - (long) DATE::Traits::base;
    if (in_range((long) DATE::MIN.get_offset(), 
                 offset, 
                 (long) DATE::MAX.get_offset()))
      return DATE((Offset) offset);
    else
      throw DateRangeError();
  }
  else
    throw InvalidDateError();
}

/*
 * Creates a date by parsing an ISO date.
 *
 * Throws <DateFormatError> if the date is not formatted correctly.
 * Throws <InvalidDateError> if the year, month, and day are invalid.
 * Throws <DateRangeError> if the date is out of range.
 */
template<class DATE>
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
 * Creates a date from a (`DATE`-specific) date offset.
 *
 * Throws <DateRangeError> if the offset is not in range for `DATE`.
 */
template<class DATE>
inline DATE
from_offset(
  typename DATE::Offset const offset)
{
  if (DATE::offset_is_valid(offset))
    return DATE(offset);
  else
    throw DateRangeError();
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


template<class DATE>
inline DATE
from_ymd(
  YmdDate const& date) 
{
  return from_ymd<DATE>(date.year, date.month, date.day);
}


/*
 * Creates a date from a YMDI.
 *
 * Throws <InvalidDateError> if the YMDI is invalid.
 * Throws <DateRangeError> if the YMDI is out of range.
 */
template<class DATE>
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
  return cron::get_weekday(date.get_datenum());
}


template<class DATE>
inline WeekDate 
get_week_date(
  DATE const date)
{ 
  ensure_valid(date);
  return cron::datenum_to_week_date(date.get_datenum());
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
  return cron::datenum_to_ymdi(date.get_datenum()); 
}


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
  return (int) date0.get_offset() - date1.get_offset();
}


//------------------------------------------------------------------------------

}  // namespace date
}  // namespace cron

