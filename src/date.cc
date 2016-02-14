#include <cstdlib>
#include <cctype>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>

#include "exc.hh"
#include "cron/date.hh"

using std::string;

namespace alxs {
namespace cron {

//------------------------------------------------------------------------------

OrdinalDateParts
datenum_to_ordinal_date_parts(
  Datenum const datenum)
{
  if (!datenum_is_valid(datenum))
    return OrdinalDateParts::get_invalid();

  // Compute the 400-year leap cycle and remainder; count from year 1.
  Year year = 1 + 400 * (datenum / 146097);
  uint32_t rem = datenum % 146097;

  // Adjust for the 100-year leap cycle and remainder.
  if (rem == 146096) {
    year += 300;
    rem = 36524;
  }
  else {
    year += 100 * (rem / 36524);
    rem %= 36524;
  }

  // Adjust for the 4-year leap cycle and remainder.
  year += 4 * (rem / 1461);
  rem %= 1461;

  // Compute the one-year cycle and remainder.
  // FIXME: Possibly wrong.  Validate carefully.
  if (rem == 1460) {
    year += 3;
    rem = 365;
  }
  else {
    year += rem / 365;
    rem %= 365;
  }

  return {.year = year, .ordinal = (Ordinal) rem};
}


DateParts
datenum_to_parts(
  Datenum const datenum,
  OrdinalDateParts const& ordinal_parts)
{
  if (!datenum_is_valid(datenum))
    return DateParts::get_invalid();

  auto const year       = ordinal_parts.year;
  auto       ordinal    = ordinal_parts.ordinal;
  auto const leap       = is_leap_year(year);

  Month month;
  Day day;
  if (ordinal < 31) {
    month = 0;
    day = ordinal;
  } 
  else if (ordinal < 59 || (leap && ordinal == 59)) {
    month = 1;
    day = ordinal - 31;
  }
  else {
    if (leap)
      --ordinal;
    if (ordinal < 90) {
      month = 2;
      day = ordinal - 59;
    }
    else if (ordinal < 120) {
      month = 3;
      day = ordinal - 90;
    }
    else if (ordinal < 151) {
      month = 4;
      day = ordinal - 120;
    }
    else if (ordinal < 181) {
      month = 5;
      day = ordinal - 151;
    }
    else if (ordinal < 212) {
      month = 6;
      day = ordinal - 181;
    }
    else if (ordinal < 243) {
      month = 7;
      day = ordinal - 212;
    }
    else if (ordinal < 273) {
      month = 8;
      day = ordinal - 243;
    }
    else if (ordinal < 304) {
      month = 9;
      day = ordinal - 273;
    }
    else if (ordinal < 334) {
      month = 10;
      day = ordinal - 304;
    }
    else {
      month = 11;
      day = ordinal - 334;
    }
  }

  return {.year = year, .month = month, .day = day};
}


WeekDateParts
datenum_to_week_date_parts(
  Datenum const datenum,
  OrdinalDateParts const& ordinal_parts,
  DateParts const& parts)
{
  if (!datenum_is_valid(datenum))
    return WeekDateParts::get_invalid();

  Year week_year;
  Week week;
  Weekday const weekday = get_weekday(datenum);

  // The week number is the week number of the nearest Thursday.
  int16_t const thursday = ordinal_parts.ordinal + THURSDAY - weekday;
  if (thursday < 0) {
    // The nearest Thursday is part of the previous week year.
    week_year = ordinal_parts.year - 1;
    // Calculate the week number of the previous December 31.  This calculation
    // relies on the fact that in this case, the previous December 31 must be:
    //   - a Thursday, in week 52
    //   - a Friday, in week 52 of a leap year or week 51 otherwise, 
    //   - a Saturday, in week 51.
    Weekday const dec31_weekday = weekday - parts.day - 1;
    week = 
      (dec31_weekday == THURSDAY
       || (dec31_weekday == FRIDAY && is_leap_year(week_year)))
      ? 52 : 51;
  }
  else if (thursday >= 365 && (thursday >= 366 || ! is_leap_year(parts.year))) {
    // The nearest Thursday is part of the next week year.
    week_year = parts.year + 1;
    week = 0;
  }
  else {
    week_year = parts.year;
    // Just count Thursdays.
    week = thursday / 7;
  }

  return {.week_year = week_year, .week = week, .weekday = weekday};
}


DateParts 
iso_parse(
  std::string const& text)
{
  if (text.length() == 10
      && isdigit(text[0])
      && isdigit(text[1])
      && isdigit(text[2])
      && isdigit(text[3])
      && text[4] == '-'
      && isdigit(text[5])
      && isdigit(text[6])
      && text[7] == '-'
      && isdigit(text[8])
      && isdigit(text[9])) {
    DateParts parts;
    parts.year  = atoi(text.substr(0, 4).c_str());
    parts.month = atoi(text.substr(5, 2).c_str()) - 1;
    parts.day   = atoi(text.substr(8, 2).c_str()) - 1;
    if (ymd_is_valid(parts.year, parts.month, parts.day))
      return parts;
    else
      throw ValueError("invalid date");
  }
  else
    throw ValueError("not ISO date format");
}


//------------------------------------------------------------------------------

}  // namespace cron
}  // namespace alxs

