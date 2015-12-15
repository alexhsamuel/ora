#include <cstdlib>
#include <cctype>
#include <iomanip>
#include <memory>
#include <sstream>

#include "exc.hh"
#include "cron/date.hh"

using std::string;

namespace alxs {
namespace cron {

//------------------------------------------------------------------------------

DateParts
datenum_to_parts(
  Datenum datenum)
{
  DateParts parts;

  if (! datenum_is_valid(datenum)) 
    return DateParts::get_invalid();

  // Shift forward to the basis year 1200.  We do this first to keep the
  // following divisions positive.
  datenum += (1200 / 400) * 146097;
  // Compute the 400-year cycle and remainder.
  parts.year = 400 * (datenum / 146097);
  uint32_t rem = datenum % 146097;

  // Compute the 100-year cycle and remainder.
  if (rem == 146096) {
    parts.year += 300;
    rem = 36524;
  }
  else {
    parts.year += 100 * (rem / 36524);
    rem %= 36524;
  }

  // Compute the 4-year cycle and remainder.
  parts.year += 4 * (rem / 1461);
  rem %= 1461;

  // Compute the one-year cycle and remainder.
  if (rem == 1460) {
    parts.year += 3;
    rem = 365;
  }
  else {
    parts.year += rem / 365;
    rem %= 365;
  }

  // Compute month and date shifting from March 1.
  if (rem < 306) {
    // March - December.
    parts.ordinal = rem + 59;
    if      (rem <  31) { 
      parts.month = 2;
      parts.day = rem -   0;
    }
    else if (rem <  61) {
      parts.month = 3;
      parts.day = rem -  31;
    }
    else if (rem <  92) {
      parts.month = 4;
      parts.day = rem -  61;
    }
    else if (rem < 122) {
      parts.month = 5;
      parts.day = rem -  92;
    }
    else if (rem < 153) {
      parts.month = 6;
      parts.day = rem - 122;
    }
    else if (rem < 184) {
      parts.month = 7;
      parts.day = rem - 153;
    }
    else if (rem < 214) {
      parts.month = 8;
      parts.day = rem - 184;
    }
    else if (rem < 245) {
      parts.month = 9;
      parts.day = rem - 214;
    }
    else if (rem < 275) {
      parts.month = 10;
      parts.day = rem - 245;
    }
    else {
      parts.month = 11;
      parts.day = rem - 275;
    }
  }
  else {
    // January - February.
    parts.year++;
    parts.ordinal = rem - 306;
    if (rem < 337) {
      parts.month = 0;
      parts.day = rem - 306;
    }
    else {
      parts.month = 1;
      parts.day = rem - 337;
    }
  }

  // 1200 March 1 is a Wednesday.
  parts.weekday = get_weekday(datenum);

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


DateParts iso_parse(
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

