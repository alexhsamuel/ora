#pragma GCC diagnostic ignored "-Wparentheses"

#include <cstdint>

#include "ora/date_math.hh"
#include "ora/format.hh"
#include "ora/lib/math.hh"
#include "ora/parse.hh"

namespace ora {

using namespace ora::lib;
using std::string;

//------------------------------------------------------------------------------
// Helpers
//------------------------------------------------------------------------------

namespace {

#define TRY(stmt) do { if (!(stmt)) return false; } while(false)

/*
 * Skips over formatting modifiers; see `parse_modifiers`.
 */
inline bool
skip_modifiers(
  char const*& p)
{
  bool decimal = false;

  for (; *p != 0; ++p)
    switch (*p) {
    case '.':
      if (decimal)
        // Two decimal points.
        return false;
      else
        decimal = true;
      break;

    case '0': case '1': case '2': case '3': case '4':
    case '5': case '6': case '7': case '8': case '9':
    case '^': case '_': case '~':
      break;

    case '#':
      if (*++p == 0)
        // Must be followed by another character.
        return false;
      break;

    default:
      return true;
    }

  // Unrechable.
  return false;
}


template<size_t MAX_DIGITS, bool FIXED=false>
inline int
parse_unsigned(
  char const*& p)
{
  int val;

  if (isdigit(*p)) 
    val = *p++ - '0';
  else
    return -1;

  for (size_t i = 0; i < MAX_DIGITS - 1; ++i)
    if (isdigit(*p))
      val = val * 10 + (*p++ - '0');
    else
      return FIXED ? -1 : val;

  return val;
}


inline bool
parse_day(
  char const*& s,
  Day& day)
{
  auto const i = parse_unsigned<2>(s);
  if (day_is_valid(i)) {
    day = i;
    return true;
  }
  else
    return false;
}


inline bool
parse_hour(
  char const*& s,
  Hour& hour)
{
  auto const i = parse_unsigned<2>(s);
  if (hour_is_valid(i)) {
    hour = i;
    return true;
  }
  else
    return false;
}


inline bool
parse_minute(
  char const*& s,
  Minute& minute)
{
  auto const i = parse_unsigned<2, true>(s);
  if (minute_is_valid(i)) {
    minute = i;
    return true;
  }
  else
    return false;
}


inline bool
parse_month(
  char const*& s,
  Month& month)
{
  auto const i = parse_unsigned<2>(s);
  if (month_is_valid(i)) {
    month = i;
    return true;
  }
  else
    return false;
}


inline bool
parse_ordinal(
  char const*& s,
  Ordinal& ordinal)
{
  auto const i = parse_unsigned<3>(s);
  if (ordinal_is_valid(i)) {
    ordinal = i;
    return true;
  }
  else
    return false;
}


inline bool
parse_second(
  char const*& s,
  Second& second)
{
  // FIXME: Parse factional seconds.
  auto const i = parse_unsigned<2, true>(s);
  if (second_is_valid(i)) {
    second = i;
    return true;
  }
  else
    return false;
}


inline bool
parse_two_digit_year(
  char const*& s,
  Year& year)
{
  auto const i = parse_unsigned<2>(s);
  if (0 <= i && i < 100) {
    year = infer_two_digit_year(i);
    return true;
  }
  else
    return false;
}


inline bool
parse_weekday_iso(
  char const*& s,
  Weekday& weekday)
{
  auto const i = parse_unsigned<1>(s);
  if (ora::weekday::ENCODING_ISO::is_valid(i)) {
    weekday = ora::weekday::ENCODING_ISO::decode(i);
    return true;
  }
  else
    return false;
}


inline bool
parse_week(
  char const*& s,
  Week& week)
{
  auto const i = parse_unsigned<2>(s);
  if (week_is_valid(i)) {
    week = i;
    return true;
  }
  else
    return false;
}


inline bool
parse_weekday_unix(
  char const*& s,
  Weekday& weekday)
{
  auto const i = parse_unsigned<1>(s);
  if (ora::weekday::ENCODING_CRON::is_valid(i)) {
    weekday = ora::weekday::ENCODING_CRON::decode(i);
    return true;
  }
  else
    return false;
}


inline bool
parse_year(
  char const*& s,
  Year& year)
{
  auto const i = parse_unsigned<4>(s);
  if (year_is_valid(i)) {
    year = i;
    return true;
  }
  else
    return false;
}


inline bool
parse_iso_date(
  char const*& s,
  YmdDate& date,
  bool compact=false)
{
  TRY(parse_year(s, date.year));
  if (!compact) {
    if (*s != '-')
      return false;
    ++s;
  }
  TRY(parse_month(s, date.month));
  if (!compact) {
    if (*s != '-')
      return false;
    ++s;
  }
  TRY(parse_day(s, date.day));
  return true;
}


}  // anonymous namespace


//------------------------------------------------------------------------------
// Dates
//------------------------------------------------------------------------------

namespace date {

bool
parse_date_parts(
  char const*& p,
  char const*& s,
  FullDate& parts)
{
  while (true)
    if (*p == 0 && *s == 0)
      // Completed successfully.
      return true;
    else if (*p == '%') {
      ++p;
      if (*p == '%')
        // Literal '%'.
        if (*s == '%') {
          ++p;
          ++s;
          continue;
        }
        else
          // Didn't match %.
          return false;

      skip_modifiers(p);

      switch (*p) {
      case 'A': TRY(parse_weekday_name(s, parts.week_date.weekday)); break;
      case 'a': TRY(parse_weekday_abbr(s, parts.week_date.weekday)); break;
      case 'B': TRY(parse_month_name(s, parts.ymd_date.month)); break;
      case 'b': TRY(parse_month_abbr(s, parts.ymd_date.month)); break;
      case 'D': TRY(parse_iso_date(s, parts.ymd_date)); break;
      case 'd': TRY(parse_day(s, parts.ymd_date.day)); break;
      case 'G': TRY(parse_year(s, parts.week_date.week_year)); break;
      case 'g': TRY(parse_two_digit_year(s, parts.week_date.week_year)); break;
      case 'j': TRY(parse_ordinal(s, parts.ordinal_date.ordinal)); break;
      case 'm': TRY(parse_month(s, parts.ymd_date.month)); break;
      case 'u': TRY(parse_weekday_iso(s, parts.week_date.weekday)); break;
      case 'V': TRY(parse_week(s, parts.week_date.week)); break;
      case 'w': TRY(parse_weekday_unix(s, parts.week_date.weekday)); break;
      case 'Y': TRY(parse_year(s, parts.ymd_date.year)); 
                parts.ordinal_date.year = parts.ymd_date.year; break;
      case 'y': TRY(parse_two_digit_year(s, parts.ymd_date.year)); break;

      default:
        return false;
      }
      ++p;
    }
    else if (*p == *s) {
      ++p;
      ++s;
    }
    else
      return false;
}


}  // namespace date


//------------------------------------------------------------------------------

namespace daytime {

struct ParseExtra
{
  Hour hour_12 = HOUR_INVALID;
  int am_pm = 0;
  int usec = -1;
};


bool parse_daytime_parts(
  char const*& p,
  char const*& s,
  HmsDaytime& parts)
{
  // Second may be omitted.
  parts.second = 0;

  ParseExtra extra;

  while (true)
    if (*p == 0 && *s == 0) {
      // Completed successfully.
      if (   parts.hour == HOUR_INVALID 
          && extra.hour_12 != HOUR_INVALID 
          && extra.am_pm != -1)
        parts.hour = 
            (extra.hour_12 == 12 ? 0 : extra.hour_12) 
          + (extra.am_pm == 1 ? 12 : 0);
      if (extra.usec != -1 && parts.second != SECOND_INVALID)
        parts.second = int(parts.second) + extra.usec * 1e-6;
      return true;
    }
    else if (*p == '%') {
      ++p;
      if (*p == '%')
        // Literal '%'.
        if (*s == '%') {
          ++p;
          ++s;
          continue;
        }
        else
          // Didn't match %.
          return false;

      skip_modifiers(p);

      switch (*p) {
      case 'f': {
        auto const last_s = s;
        auto i = parse_unsigned<6>(s);
        if (i == -1) 
          return false;
        else 
          // If fewer than six digits, scale to zero-pad on the right.
          extra.usec = i * ora::lib::pow10(6 - (s - last_s));
      }; break;

      case 'H': TRY(parse_hour(s, parts.hour)); break;

      case 'I': {
        auto const i = parse_unsigned<2>(s);
        if (1 <= i && i <= 12)
          extra.hour_12 = i;
        else
          return false;
      }; break;

      case 'M': TRY(parse_minute(s, parts.minute)); break;

      case 'p':
        if (   (*s == 'a' || *s == 'A')
            && (*(s + 1) == 'm' || *(s + 1) == 'M'))
          extra.am_pm = 0;
        else if (   (*s == 'p' || *s == 'P')
                 && (*(s + 1) == 'm' || *(s + 1) == 'M'))
          extra.am_pm = 1;
        else
          return false;
        s += 2;
        break;

      case 'S': TRY(parse_second(s, parts.second)); break;

      default:
        return false;
      }
      ++p;
    }
    else if (*p == *s) {
      ++p;
      ++s;
    }
    else
      return false;
}


}  // namespace daytime


//------------------------------------------------------------------------------

namespace time {


}  // namespace time

//------------------------------------------------------------------------------

}  // namespace ora

