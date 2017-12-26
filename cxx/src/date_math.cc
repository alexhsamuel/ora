#include <cstdlib>
#include <cctype>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>

#include "aslib/exc.hh"
#include "ora.hh"

namespace ora {

using std::string;

using namespace aslib;

//------------------------------------------------------------------------------

OrdinalDate
datenum_to_ordinal_date(
  Datenum const datenum)
  noexcept
{
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

  return OrdinalDate{year, (Ordinal) (days + 1)};
}


extern YmdDate
datenum_to_ymd(
  Datenum const datenum,
  OrdinalDate const ordinal_date)
  noexcept
{
  auto const leap = is_leap_year(ordinal_date.year);

  Ordinal days = ordinal_date.ordinal;
  Month month;
  Day day;

  if (days < 32) {
    month = 1;
    day = days;
  } 
  else if (days < 60 || (leap && days == 60)) {
    month = 2;
    day = days - 31;
  }
  else {
    if (leap)
      --days;
    if (days < 91) {
      month = 3;
      day = days - 59;
    }
    else if (days < 121) {
      month = 4;
      day = days - 90;
    }
    else if (days < 152) {
      month = 5;
      day = days - 120;
    }
    else if (days < 182) {
      month = 6;
      day = days - 151;
    }
    else if (days < 213) {
      month = 7;
      day = days - 181;
    }
    else if (days < 244) {
      month = 8;
      day = days - 212;
    }
    else if (days < 274) {
      month = 9;
      day = days - 243;
    }
    else if (days < 305) {
      month = 10;
      day = days - 273;
    }
    else if (days < 335) {
      month = 11;
      day = days - 304;
    }
    else {
      month = 12;
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
  noexcept
{
  auto const year = ordinal_date.year;
  auto const days = ordinal_date.ordinal - 1;

  Year week_year;
  Week week;

  // The week number is the week number of the nearest Thursday.
  int16_t const thursday = days + THURSDAY - weekday;
  if (thursday < 0) {
    // The nearest Thursday is part of the previous week year.
    week_year = year - 1;
    // Calculate the week number of the previous December 31.  This calculation
    // relies on the fact that in this case, the previous December 31 must be:
    //   - a Thursday, in week 52
    //   - a Friday, in week 52 of a leap year or week 51 otherwise, 
    //   - a Saturday, in week 51.
    Weekday const dec31_weekday = weekday - days - 1;
    week = 
         dec31_weekday == THURSDAY
      || (dec31_weekday == FRIDAY && is_leap_year(week_year))
      ? 53 : 52;
  }
  else if (thursday >= 365 + is_leap_year(year)) {
    // The nearest Thursday is part of the next week year.
    week_year = year + 1;
    week = 1;
  }
  else {
    week_year = year;
    // Just count Thursdays.
    week = thursday / 7 + 1;
  }

  return WeekDate{week_year, week, weekday};
}


YmdDate
parse_iso_date(
  std::string const& text)
  noexcept
{
  auto const len = text.length();
  if (
       len == 8
    && isdigit(text[0])
    && isdigit(text[1])
    && isdigit(text[2])
    && isdigit(text[3])
    && isdigit(text[4])
    && isdigit(text[5])
    && isdigit(text[6])
    && isdigit(text[7])) 
    return {
      (Year)   atoi(text.substr(0, 4).c_str()),
      (Month) (atoi(text.substr(4, 2).c_str())),
      (Day)   (atoi(text.substr(6, 2).c_str())),
    };
  else if (
       len == 10
    && isdigit(text[0])
    && isdigit(text[1])
    && isdigit(text[2])
    && isdigit(text[3])
    && text[4] == '-'
    && isdigit(text[5])
    && isdigit(text[6])
    && text[7] == '-'
    && isdigit(text[8])
    && isdigit(text[9])) 
    return {
      (Year)   atoi(text.substr(0, 4).c_str()),
      (Month) (atoi(text.substr(5, 2).c_str())),
      (Day)   (atoi(text.substr(8, 2).c_str())),
    };
  else
    return YmdDate{};  // invalid
}


//------------------------------------------------------------------------------

FullDate
datenum_to_full_date(
  Datenum const datenum)
  noexcept
{
  if (! datenum_is_valid(datenum)) 
    return {};

  auto const ord = datenum_to_ordinal_date(datenum);
  auto const ymd = datenum_to_ymd(datenum, ord);
  auto const wdy = get_weekday(datenum);
  auto const wdt = datenum_to_week_date(datenum, ord, wdy);

  return {ord, ymd, wdt};
}


//------------------------------------------------------------------------------

}  // namespace ora


