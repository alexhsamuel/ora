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

OrdinalDate
datenum_to_ordinal_date(
  Datenum const datenum)
{
  if (! datenum_is_valid(datenum)) 
    return OrdinalDate::get_invalid();

  // Compute the 400-year leap cycle and remainder; count from year 1.
  Year year = 1 + 400 * (datenum / 146097);
  uint32_t days = datenum % 146097;

  // Adjust for the 100-year leap cycle and remainder.
  if (days == 146096) {
    year += 300;
    days = 36524;
  }
  else {
    year += 100 * (days / 36524);
    days %= 36524;
  }

  // Adjust for the 4-year leap cycle and remainder.
  year += 4 * (days / 1461);
  days %= 1461;

  // Compute the one-year cycle and remainder.
  // FIXME: Probably wrong.  Validate carefully.
  if (days == 1460) {
    year += 3;
    days = 365;
  }
  else {
    year += days / 365;
    days %= 365;
  }

  return OrdinalDate{year, (Ordinal) days};
}


extern YmdDate
datenum_to_ymd(
  Datenum const datenum,
  OrdinalDate const ordinal_date)
{
  auto const leap = is_leap_year(ordinal_date.year);

  Ordinal days = ordinal_date.ordinal;
  Month month;
  Day day;

  if (days < 31) {
    month = 0;
    day = days;
  } 
  else if (days < 59 || (leap && days == 59)) {
    month = 1;
    day = days - 31;
  }
  else {
    if (leap)
      --days;
    if (days < 90) {
      month = 2;
      day = days - 59;
    }
    else if (days < 120) {
      month = 3;
      day = days - 90;
    }
    else if (days < 151) {
      month = 4;
      day = days - 120;
    }
    else if (days < 181) {
      month = 5;
      day = days - 151;
    }
    else if (days < 212) {
      month = 6;
      day = days - 181;
    }
    else if (days < 243) {
      month = 7;
      day = days - 212;
    }
    else if (days < 273) {
      month = 8;
      day = days - 243;
    }
    else if (days < 304) {
      month = 9;
      day = days - 273;
    }
    else if (days < 334) {
      month = 10;
      day = days - 304;
    }
    else {
      month = 11;
      day = days - 334;
    }
  }

  return YmdDate{ordinal_date.year, month, day};
}


extern WeekDate
datenum_to_week_date(
  Datenum const datenum,
  OrdinalDate const ordinal_date,
  Weekday const weekday)
{
  auto const year    = ordinal_date.year;
  auto const ordinal = ordinal_date.ordinal;

  Year week_year;
  Week week;

  // The week number is the week number of the nearest Thursday.
  int16_t const thursday = ordinal + THURSDAY - weekday;
  if (thursday < 0) {
    // The nearest Thursday is part of the previous week year.
    week_year = year - 1;
    // Calculate the week number of the previous December 31.  This calculation
    // relies on the fact that in this case, the previous December 31 must be:
    //   - a Thursday, in week 52
    //   - a Friday, in week 52 of a leap year or week 51 otherwise, 
    //   - a Saturday, in week 51.
    Weekday const dec31_weekday = weekday - ordinal - 1;
    week = 
         dec31_weekday == THURSDAY
      || (dec31_weekday == FRIDAY && is_leap_year(week_year))
      ? 52 : 51;
  }
  else if (thursday >= 365 + is_leap_year(year)) {
    // The nearest Thursday is part of the next week year.
    week_year = year + 1;
    week = 0;
  }
  else {
    week_year = year;
    // Just count Thursdays.
    week = thursday / 7;
  }

  return WeekDate{week_year, week, weekday};
}


//------------------------------------------------------------------------------

// FIXME: Remove, eventually.

DateParts
datenum_to_parts(
  Datenum const datenum)
{
  DateParts parts;

  if (! datenum_is_valid(datenum)) 
    return DateParts::get_invalid();

  // Compute the 400-year leap cycle and remainder; count from year 1.
  parts.year = 1 + 400 * (datenum / 146097);
  uint32_t rem = datenum % 146097;

  // Adjust for the 100-year leap cycle and remainder.
  if (rem == 146096) {
    parts.year += 300;
    rem = 36524;
  }
  else {
    parts.year += 100 * (rem / 36524);
    rem %= 36524;
  }

  // Adjust for the 4-year leap cycle and remainder.
  parts.year += 4 * (rem / 1461);
  rem %= 1461;

  // Compute the one-year cycle and remainder.
  // FIXME: Probably wrong.  Validate carefully.
  if (rem == 1460) {
    parts.year += 3;
    rem = 365;
  }
  else {
    parts.year += rem / 365;
    rem %= 365;
  }

  parts.ordinal = rem;
  parts.weekday = get_weekday(datenum);

  auto const leap = is_leap_year(parts.year);

  if (rem < 31) {
    parts.month = 0;
    parts.day = rem;
  } 
  else if (rem < 59 || (leap && rem == 59)) {
    parts.month = 1;
    parts.day = rem - 31;
  }
  else {
    if (leap)
      --rem;
    if (rem < 90) {
      parts.month = 2;
      parts.day = rem - 59;
    }
    else if (rem < 120) {
      parts.month = 3;
      parts.day = rem - 90;
    }
    else if (rem < 151) {
      parts.month = 4;
      parts.day = rem - 120;
    }
    else if (rem < 181) {
      parts.month = 5;
      parts.day = rem - 151;
    }
    else if (rem < 212) {
      parts.month = 6;
      parts.day = rem - 181;
    }
    else if (rem < 243) {
      parts.month = 7;
      parts.day = rem - 212;
    }
    else if (rem < 273) {
      parts.month = 8;
      parts.day = rem - 243;
    }
    else if (rem < 304) {
      parts.month = 9;
      parts.day = rem - 273;
    }
    else if (rem < 334) {
      parts.month = 10;
      parts.day = rem - 304;
    }
    else {
      parts.month = 11;
      parts.day = rem - 334;
    }
  }

  // The week number is the week number of the nearest Thursday.
  int16_t const thursday = parts.ordinal + THURSDAY - parts.weekday;
  if (thursday < 0) {
    // The nearest Thursday is part of the previous week year.
    parts.week_year = parts.year - 1;
    // Calculate the week number of the previous December 31.  This calculation
    // relies on the fact that in this case, the previous December 31 must be:
    //   - a Thursday, in week 52
    //   - a Friday, in week 52 of a leap year or week 51 otherwise, 
    //   - a Saturday, in week 51.
    Weekday const dec31_weekday = parts.weekday - parts.day - 1;
    parts.week = 
      (dec31_weekday == THURSDAY
       || (dec31_weekday == FRIDAY && is_leap_year(parts.week_year)))
      ? 52 : 51;
  }
  else if (thursday >= 365 && (thursday >= 366 || ! is_leap_year(parts.year))) {
    // The nearest Thursday is part of the next week year.
    parts.week_year = parts.year + 1;
    parts.week = 0;
  }
  else {
    parts.week_year = parts.year;
    // Just count Thursdays.
    parts.week = thursday / 7;
  }

  return parts;
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

