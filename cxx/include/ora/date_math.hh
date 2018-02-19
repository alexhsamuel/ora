/*
 * Basic date calculations.
 *
 * Some of these functions do _not_ check the validity of their arguments (other
 * than the *_is_valid() functions); their results are undefined for invalid
 * values.
 */

#pragma once

#include "ora/lib/exc.hh"
#include "ora/lib/math.hh"
#include "ora/types.hh"

// FIXME: Place in ora::date::math.

namespace ora {

using namespace ora::lib;

//------------------------------------------------------------------------------
// Declarations
//------------------------------------------------------------------------------

/*
 * Returns ordinal date parts for a date.
 */
extern OrdinalDate datenum_to_ordinal_date(Datenum) noexcept;

/*
 * Returns YMD date parts for a date.
 */
extern YmdDate datenum_to_ymd(Datenum, OrdinalDate) noexcept;

/*
 * Returns week date parts for a date.
 */
extern WeekDate datenum_to_week_date(Datenum, OrdinalDate, Weekday) noexcept;

/*
 * Returns date parts for a date.
 */
extern FullDate datenum_to_full_date(Datenum) noexcept;

/*
 * Parses an ISO-8601 extended date ("YYYY-MM-DD" format) into parts.
 */
extern YmdDate parse_iso_date(std::string const&) noexcept;

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
days_in_year(
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
    && in_range(ORDINAL_MIN, ordinal, days_in_year(year));
}


/*
 * Returns the number of days in a month.
 *
 * The year is required to account for leap years.
 */
inline Day constexpr
days_in_month(
  Year const year,
  Month const month)
{
  return 
      month ==  4 || month ==  6 || month ==  9 || month == 11 ? 30
    : month == 2 ? (is_leap_year(year) ? 29 : 28)
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
    && in_range(DAY_MIN, day, days_in_month(year, month));
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
 * Infers the full year for a two-digit year.
 *
 * This calculation is arbitrary.
 */
inline Year constexpr
infer_two_digit_year(
  int const val)
{
  return 
      val <   0 ? YEAR_INVALID
    : val <  69 ? 2000 + val
    : val < 100 ? 1900 + val
    :             YEAR_INVALID;
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
  return jan1_datenum(year) + ordinal - 1;
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
      (month == 1) ?    0
    : (month == 2) ?   31
    : (
         (month ==  3) ?  59
       : (month ==  4) ?  90
       : (month ==  5) ? 120
       : (month ==  6) ? 151
       : (month ==  7) ? 181
       : (month ==  8) ? 212
       : (month ==  9) ? 243
       : (month == 10) ? 273
       : (month == 11) ? 304
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
    + day - 1;
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
    && in_range(WEEK_MIN, week, weeks_in_week_year(week_year));
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
    + (week - 1) * 7                    // Add the week offset.
    + weekday;                          // Add the weekday offset.
}


/*
 * Returns YMD date parts for a date.
 */
inline YmdDate
datenum_to_ymd(
  Datenum const datenum)
  noexcept
{
  return datenum_to_ymd(datenum, datenum_to_ordinal_date(datenum));
}


/*
 * Returns week date parts for a date.
 */
inline WeekDate 
datenum_to_week_date(
  Datenum const datenum)
  noexcept
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
       in_interval(YMDI_MIN, ymdi, YMDI_END)
    && ymd_is_valid(ymdi / 10000, ymdi / 100 % 100, ymdi % 100);
}


/*
 * Computes the date from a YMDI.
 */
inline Datenum 
ymdi_to_datenum(
  int const ymdi)
  noexcept
{
  return ymd_to_datenum(ymdi / 10000, ymdi / 100 % 100, ymdi % 100);
}


inline int 
datenum_to_ymdi(
  Datenum const datenum)
  noexcept
{
  auto const ymd = datenum_to_ymd(datenum);
  return 10000 * ymd.year + 100 * ymd.month + ymd.day;
}


/*
 * Computes datenum from parts.
 *
 * Uses components of FullDate in this order of precedence:
 * - YMD date, if fully specified.
 * - Week date, if fully specified.
 * - Ordinal date, if fully specified.
 */
inline Datenum
parts_to_datenum(
  FullDate const parts)
{
  if (ymd_is_valid(
      parts.ymd_date.year, parts.ymd_date.month, parts.ymd_date.day))
    return ymd_to_datenum(
      parts.ymd_date.year, parts.ymd_date.month, parts.ymd_date.day);

  else if (week_date_is_valid(
      parts.week_date.week_year, parts.week_date.week, parts.week_date.weekday))
    return week_date_to_datenum(
      parts.week_date.week_year, parts.week_date.week, parts.week_date.weekday);

  else if (ordinal_date_is_valid(
      parts.ordinal_date.year, parts.ordinal_date.ordinal))
    return ordinal_date_to_datenum(
      parts.ordinal_date.year, parts.ordinal_date.ordinal);

  else
    return DATENUM_INVALID;
}


//------------------------------------------------------------------------------

namespace weekday {

/*
 * Encoding convention of weekdays to integers.
 *
 * A convention is parameterized by `MIN`, the lowest-numbered weekday, and
 * `VAL`, the integer value of `MIN`.
 *
 * The other weekdays are numbered sequentially up to `VAL` + 6.
 *
 * For example, a weekday numbering convention that assigns Sunday = -1,
 * Monday = 0, ..., Saturday = 5, is given by,
 *
 *     Convention<SUNDAY, -1>
 *
 */
template<Weekday MIN, int VAL>
class Convention
{
public:

  static constexpr bool 
  is_valid(
    int const c)
  {
    return in_interval(VAL, c, VAL + 7);
  }

  static constexpr int 
  encode(
    Weekday const c)
  {
    return 
      VAL + (c < MIN ? (int) c - (MIN - 7) : (int) c - MIN);
  }

  static constexpr Weekday
  decode(
    int const c)
  {
    return c - VAL + MIN >= 7 ? c - VAL + (MIN - 7) : c - VAL + MIN;
  }

private:

  // May not be instantiated.
  Convention()                  = delete;
  Convention(Convention const&) = delete;
  Convention(Convention&&)      = delete;
  ~Convention()                 = delete;

};

/*
 * Cron's own weekday convention.
 */
using ENCODING_CRON     = Convention<MONDAY, 0>;

/* 
 * ISO 8601 weekday convention.
 */
using ENCODING_ISO      = Convention<MONDAY, 1>;

/*
 * The C library's (i.e. `struct tm`) convention.
 */
using ENCODING_CLIB     = Convention<SUNDAY, 0>;

}  // namespace weekday


//------------------------------------------------------------------------------

}  // namespace ora

