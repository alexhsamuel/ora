/*
 * Basic date calculations.
 *
 * These functions do _not_ check the validity of their arguments (other than
 * the *_is_valid() functions), and their results are undefined for invalid
 * values.
 */

#pragma once

#include "cron/math.hh"
#include "cron/types.hh"
#include "exc.hh"

namespace alxs {
namespace cron {

//------------------------------------------------------------------------------
// Declarations
//------------------------------------------------------------------------------

/*
 * Returns ordinal date parts for a date.
 */
extern OrdinalDate datenum_to_ordinal_date(Datenum);

/*
 * Returns YMD date parts for a date.
 */
extern YmdDate datenum_to_ymd(Datenum, OrdinalDate);

/*
 * Returns week date parts for a date.
 */
extern WeekDate datenum_to_week_date(Datenum, OrdinalDate, Weekday);

/*
 * Returns date parts for a date.
 */
extern DateParts datenum_to_parts(Datenum);

/*
 * Parses an ISO-8601 extended date ("YYYY-MM-DD" format) into parts.
 *
 * FIXME: Remove this and use DateFormat.
 */
extern DateParts iso_parse(std::string const& text);  

//------------------------------------------------------------------------------
// Inline functions
//------------------------------------------------------------------------------

inline bool constexpr
is_leap_year(
  Year const year)
{
  return year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
}


/*
 * Returns the number of days in the year.
 */
inline Ordinal constexpr
days_per_year(
  Year const year)
{
  return is_leap_year(year) ? 366 : 365;
}


/*
 * True if `year, ordinal` form a valid ordinal date.
 */
inline bool constexpr
ordinal_date_is_valid(
  Year const year,
  Ordinal const ordinal)
{
  return 
       year_is_valid(year) 
    && in_interval(ORDINAL_MIN, ordinal, days_per_year(year));
}


/*
 * Returns the number of days in a month.
 *
 * The year is required to account for leap years.
 */
inline Day constexpr
days_per_month(
  Year const year,
  Month const month)
{
  return 
      month ==  3 || month ==  5 || month ==  8 || month == 10 ? 30
    : month == 1 ? (is_leap_year(year) ? 29 : 28)
    : 31;
}


/*
 * True if `year, month, day` form a valid date.
 */
inline bool constexpr
ymd_is_valid(
  Year const year,
  Month const month,
  Day const day)
{
  return 
       month_is_valid(month)
    && year_is_valid(year)
    && in_interval(DAY_MIN, day, days_per_month(year, month));
}


/*
 * Returns the weekday for a date.
 */
inline Weekday constexpr
get_weekday(
  Datenum const datenum)
{
  // 0001-01-01 is a Monday.
  return (MONDAY + datenum) % 7;
}


/*
 * Returns the datenum for Jan 1 of 'year'.
 */
inline Datenum constexpr
jan1_datenum(
  Year const year)
{
  return
    // An ordinary year has 365 days; count from year 1.
    365 * (year - 1)
    // Add a leap day for multiples of four; century years are not leap years
    // unless also a multiple of 400.  Subtract one from the year, since we
    // are considering Jan 1 and therefore care about previous years only.
    + (year - 1) /   4
    - (year - 1) / 100
    + (year - 1) / 400;
}


inline Datenum constexpr 
ordinal_date_to_datenum(
  Year const year,
  Ordinal const ordinal)
{
  return jan1_datenum(year) + ordinal;
}


/*
 * Returns the number of days since Jan 1 of the first day of a month.
 */
inline Datenum constexpr
get_month_offset(
  Year const year, 
  Month const month)
{
  // The cumbersome construction is required for constexpr.
  return
      (month == 0) ?    0
    : (month == 1) ?   31
    : (
         (month ==  2) ?  59
       : (month ==  3) ?  90
       : (month ==  4) ? 120
       : (month ==  5) ? 151
       : (month ==  6) ? 181
       : (month ==  7) ? 212
       : (month ==  8) ? 243
       : (month ==  9) ? 273
       : (month == 10) ? 304
       :                 334
      ) + (is_leap_year(year) ? 1 : 0);
}


inline Datenum constexpr 
ymd_to_datenum(
  Year const year,
  Month const month,
  Day const day)
{
  return
      jan1_datenum(year)
    + get_month_offset(year, month)
    + day;
}


/*
 * Returns the weekday of the first day of a year.
 */
inline Weekday constexpr
jan1_weekday(
  Year const year)
{
  return get_weekday(jan1_datenum(year));
}


/*
 * Returns the number of weeks in a week year.
 */
inline Week constexpr
weeks_in_week_year(
  Year const week_year)
{
  return
       jan1_weekday(week_year) == THURSDAY
    || (is_leap_year(week_year) && jan1_weekday(week_year) == WEDNESDAY)
    ? 53
    : 52;
}


inline bool constexpr
week_date_is_valid(
  Year const week_year,
  Week const week,
  Weekday const weekday)
{
  return
       year_is_valid(week_year)
    && weekday_is_valid(weekday)
    && in_interval(WEEK_MIN, week, weeks_in_week_year(week_year));
}


/*
 * Computes the date from a week date.
 */
inline Datenum constexpr 
week_date_to_datenum(
  Year const week_year,
  Week const week,
  Weekday const weekday)
{
  // FIXME: Validate.
  Datenum const jan1 = jan1_datenum(week_year);
  return 
      jan1                              // Start with Jan 1.
    + (10 - get_weekday(jan1)) % 7 - 3  // Adjust to start on the full week.
    + week * 7                          // Add the week offset.
    + weekday;                          // Add the weekday offset.
}


/*
 * Returns YMD date parts for a date.
 */
inline YmdDate
datenum_to_ymd(
  Datenum const datenum)
{
  return datenum_to_ymd(datenum, datenum_to_ordinal_date(datenum));
}


/*
 * Returns week date parts for a date.
 */
inline WeekDate 
datenum_to_week_date(
  Datenum const datenum)
{
  return datenum_to_week_date(
    datenum, datenum_to_ordinal_date(datenum), get_weekday(datenum));
}


/*
 * True if a YMDI is valid.
 *
 * By convention, a YMDI must be at least 10000000, so years before 1000 are
 * not representable.
 */
inline bool constexpr
ymdi_is_valid(
  int const ymdi)
{
  return 
       in_interval(YMDI_MIN, ymdi, YMDI_BOUND)
    && ymd_is_valid(ymdi / 10000, ymdi / 100 % 100 - 1, ymdi % 100 - 1);
}


/*
 * Computes the date from a YMDI.
 */
inline Datenum constexpr
ymdi_to_datenum(
  int const ymdi)
{
  return ymd_to_datenum(ymdi / 10000, ymdi / 100 % 100 - 1, ymdi % 100 - 1);
}


inline int
datenum_to_ymdi(
  Datenum const datenum)
{
  auto const ymd = datenum_to_ymd(datenum);
  return 10000 * ymd.year + 100 * (ymd.month + 1) + (ymd.day + 1);
}


//------------------------------------------------------------------------------

}  // namespace alxs
}  // namespace cron


