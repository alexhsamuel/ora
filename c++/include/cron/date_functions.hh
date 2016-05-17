#pragma once

#include "cron/date.hh"
#include "cron/date_math.hh"
#include "cron/types.hh"

namespace cron {
namespace date {

//------------------------------------------------------------------------------
// Forward declarations
//------------------------------------------------------------------------------

template<class DATE> DATE from_ymd(YmdDate const&);

//------------------------------------------------------------------------------

/*
 * Creates a date by parsing an ISO date.
 *
 * Throws <DateFormatError> if the date is not formatted correctly.
 * Throws <InvalidDateError> if the year, month, and day are invalid.
 * Throws <DateRangeError> if the date is out of range.
 */
template<class DATE>
DATE
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
DATE
from_ordinal_date(
  Year const year, 
  Ordinal const ordinal) 
{ 
  if (ordinal_date_is_valid(year, ordinal))
    return DATE::from_datenum(ordinal_date_to_datenum(year, ordinal));
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
DATE
from_week_date(
  Year const week_year,
  Week const week,
  Weekday const weekday)
{
  if (week_date_is_valid(week_year, week, weekday))
    return DATE::from_datenum(week_date_to_datenum(week_year, week, weekday));
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
DATE
from_ymd(
  Year const year, 
  Month const month, 
  Day const day) 
{
  if (ymd_is_valid(year, month, day))
    return DATE::from_datenum(ymd_to_datenum(year, month, day));
  else
    throw InvalidDateError();
}


template<class DATE>
DATE
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
DATE
from_ymdi(
  int const ymdi) 
{ 
  if (ymdi_is_valid(ymdi)) 
    return DATE::from_datenum(ymdi_to_datenum(ymdi));
  else
    throw InvalidDateError();
}


//------------------------------------------------------------------------------
// Accessors
//------------------------------------------------------------------------------

template<class DATE>
OrdinalDate 
get_ordinal_date(
  DATE const date)
{ 
  if (date.is_valid())
    return datenum_to_ordinal_date(date.get_datenum());
  else
    throw InvalidDateError();
}


template<class DATE>
YmdDate 
get_ymd(
  DATE const date)
{ 
  if (date.is_valid())
    return datenum_to_ymd(date.get_datenum()); 
  else
    throw InvalidDateError();
}


template<class DATE>
Weekday 
get_weekday(
  DATE const date)
{ 
  if (date.is_valid())
    return cron::get_weekday(date.get_datenum());
  else
    throw InvalidDateError();
}


template<class DATE>
WeekDate 
get_week_date(
  DATE const date)
{ 
  if (date.is_valid())
    return cron::datenum_to_week_date(date.get_datenum());
  else
    throw InvalidDateError();
}


template<class DATE>
int 
get_ymdi(
  DATE const date)
{ 
  if (date.is_valid())
    return cron::datenum_to_ymdi(date.get_datenum()); 
  else
    throw InvalidDateError();
}


//------------------------------------------------------------------------------

}  // namespace date
}  // namespace cron

