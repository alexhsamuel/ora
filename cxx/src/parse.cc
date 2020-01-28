#pragma GCC diagnostic ignored "-Wparentheses"

#include <cstdint>

#include "ora/date_math.hh"
#include "ora/format.hh"
#include "ora/lib/math.hh"
#include "ora/parse.hh"

namespace ora {

using namespace ora::lib;
using ora::_impl::Modifiers;

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


/*
 * Parses modifiers after % and before the type character.
 *
 * Note that width, precision, and padding are ignored when parsing.
 */
// FIXME: This should be unified with the version in format.cc.
Modifiers
parse_modifiers(
  char const*& p)
{
  Modifiers mods;
  while (true)
    switch (*p) {
    case '.':
      if (mods.decimal)
        // Already saw a decimal point in this escape.
        throw ValueError("second decimal point in escape");
      else {
        mods.decimal = true;
        p++;
      }
      break;

    case '0': case '1': case '2': case '3': case '4':
    case '5': case '6': case '7': case '8': case '9':
      // Note: width and precision not used in parsing.
      {
        int value = *p++ - '0';
        while (*p >= '0' && *p <= '9')
          value = value * 10 + (*p++ - '0');
        if (mods.decimal)
          mods.precision = value;
        else
          mods.width = value;
      }
      break;

    case '#':
      p++;
      if (*p == 0)
        throw ValueError("unterminated escape in pattern");
      mods.pad = *p++;
      break;

    case '^':
    case '_':
      mods.str_case = *p++;
      break;

    case '~':
      mods.abbreviate = true;
      p++;
      break;

    default:
      // Not a modifier.  Stop.
      return mods;

    }
}


inline bool
parse_char(
  char const*& s,
  char c)
{
  if (*s == c) {
    ++s;
    return true;
  }
  else
    return false;
}


template<size_t MAX_DIGITS, bool FIXED=false, class INT=int>
inline bool
parse_unsigned(
  char const*& s,
  INT& val)
{
  if (isdigit(*s)) 
    val = *s++ - '0';
  else
    return false;

  for (size_t i = 0; i < MAX_DIGITS - 1; ++i)
    if (isdigit(*s))
      val = val * 10 + (*s++ - '0');
    else
      return !FIXED;

  // Got all digits.
  return true;
}


template<size_t MAX_DIGITS>
inline bool
parse_signed(
  char const*& s,
  int& val)
{
  int sign = 1;
  if (*s == '-') {
    sign = -1;
    ++s;
  }
  else if (*s == '+')
    ++s;
  int abs;
  TRY(parse_unsigned<MAX_DIGITS>(s, abs));
  val = sign * abs;
  return true;
}


inline bool
parse_am_pm(
  char const*& s,
  int& am_pm)
{
  if (   (*s == 'a' || *s == 'A')
      && (*(s + 1) == 'm' || *(s + 1) == 'M'))
    am_pm = 0;
  else if (   (*s == 'p' || *s == 'P')
           && (*(s + 1) == 'm' || *(s + 1) == 'M'))
    am_pm = 1;
  else
    return false;
  s += 2;
  return true;
}


inline bool
parse_day(
  char const*& s,
  Day& day)
{
  TRY(parse_unsigned<2>(s, day));
  return day_is_valid(day);
}


inline bool
parse_hour(
  char const*& s,
  Hour& hour)
{
  TRY(parse_unsigned<2>(s, hour));
  return hour_is_valid(hour);
}


inline bool
parse_hour12(
  char const*& s,
  Hour& hour12)
{
  TRY(parse_unsigned<2>(s, hour12));
  return 1 <= hour12 && hour12 <= 12;
}


inline bool
parse_minute(
  char const*& s,
  Minute& minute)
{
  TRY((parse_unsigned<2, true>(s, minute)));
  return minute_is_valid(minute);
}


inline bool
parse_month(
  char const*& s,
  Month& month)
{
  TRY(parse_unsigned<2>(s, month));
  return month_is_valid(month);
}


inline bool
parse_ordinal(
  char const*& s,
  Ordinal& ordinal)
{
  TRY(parse_unsigned<3>(s, ordinal));
  return ordinal_is_valid(ordinal);
}


inline bool
parse_second(
  char const*& s,
  Second& second)
{
  if (*s != 0 && *(s + 1) != 0 && *(s + 2) == '.') {
    // Fractional seconds.
    // FIXME: Use a faster fractional parsing computation, or better yet work
    // entirely in terms of dayticks.
    char const* end;
    second = strtod(s, const_cast<char**>(&end));
    if (end != s && second_is_valid(second)) {
      s = end;
      return true;
    }
    else
      return false;
  }
  else {
    // Whole seconds only.
    TRY((parse_unsigned<2, true>(s, second)));
    return second_is_valid(second);
  }
}


inline bool
parse_two_digit_year(
  char const*& s,
  Year& year)
{
  int year2;
  TRY(parse_unsigned<2>(s, year2));
  if (0 <= year2 && year2 < 100) {
    year = infer_two_digit_year(year2);
    return true;
  }
  else
    return false;
}


/*
 * Parses a time zone name.
 *
 * Assumes a time zone name matches:
 *   [A-Za-z0-9/_+-]+
 */
inline bool
parse_tz_name(
  char const*& s,
  std::string& name)
{
  for (; isalnum(*s) || *s == '/' || *s == '_' || *s == '+' || *s == '-'; ++s)
    name += *s;
  return !name.empty();
}


inline bool
parse_tz_offset(
  char const*& s,
  TimeZoneOffset& tz_offset,
  bool const colon=true)
{
  int sign;
  if (*s == '+')
    sign = 1;
  else if (*s == '-')
    sign = -1;
  else
    return false;
  ++s;
  // FIXME: Accept one-digit hours for ISO-style offsets?
  int hours;
  TRY((parse_unsigned<2, true>(s, hours)));
  if (colon) 
    TRY(parse_char(s, ':'));
  int minutes;
  TRY((parse_unsigned<2, true>(s, minutes)));
  tz_offset = sign * (hours * SECS_PER_HOUR + minutes * SECS_PER_MIN);
  return true;
}


inline bool
parse_tz_offset_letter(
  char const*& s,
  TimeZoneOffset& tz_offset)
{
  auto const off = parse_time_zone_offset_letter(*s);
  if (off == TIME_ZONE_OFFSET_INVALID)
    return false;
  else {
    ++s;
    tz_offset = off;
    return true;
  }
}


inline bool
parse_tz_offset_secs(
  char const*& s,
  TimeZoneOffset& tz_offset)
{
  return parse_signed<5>(s, tz_offset);
}


inline bool
parse_usec(
  char const*& s,
  int& usec)
{
  auto const last_s = s;
  TRY(parse_unsigned<6>(s, usec));
  // If fewer than six digits, scale to zero-pad on the right.
  usec *= ora::lib::pow10(6 - (s - last_s));
  return true;
}


inline bool
parse_weekday_iso(
  char const*& s,
  Weekday& weekday)
{
  int w;
  TRY(parse_unsigned<1>(s, w));
  if (ora::weekday::ENCODING_ISO::is_valid(w)) {
    weekday = ora::weekday::ENCODING_ISO::decode(w);
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
  TRY(parse_unsigned<2>(s, week));
  return week_is_valid(week);
}


inline bool
parse_weekday_unix(
  char const*& s,
  Weekday& weekday)
{
  int w;
  TRY(parse_unsigned<1>(s, w));
  if (ora::weekday::ENCODING_CRON::is_valid(w)) {
    weekday = ora::weekday::ENCODING_CRON::decode(w);
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
  TRY(parse_unsigned<4>(s, year));
  return year_is_valid(year);
}


inline bool
parse_iso_date(
  char const*& s,
  YmdDate& date,
  bool const compact=false)
{
  TRY(parse_year(s, date.year));
  if (!compact)
    TRY(parse_char(s, '-'));
  TRY(parse_month(s, date.month));
  if (!compact)
    TRY(parse_char(s, '-'));
  TRY(parse_day(s, date.day));
  return true;
}


inline bool
parse_iso_daytime(
  char const*& s,
  HmsDaytime& hms,
  bool const compact=false)
{
  TRY((parse_unsigned<2, true>(s, hms.hour)));
  if (!hour_is_valid(hms.hour))
    return false;
  if (!compact)
    TRY(parse_char(s, ':'));
  TRY(parse_minute(s, hms.minute));
  if (!compact)
    TRY(parse_char(s, ':'));
  TRY(parse_second(s, hms.second));
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

      auto const mods = parse_modifiers(p);

      switch (*p) {
      case 'A': 
        TRY(
          mods.abbreviate
          ? parse_weekday_abbr(s, parts.week_date.weekday)
          : parse_weekday_name(s, parts.week_date.weekday));
        break;

      case 'a':
        TRY(parse_weekday_abbr(s, parts.week_date.weekday));
        break;

      case 'B':
        TRY(
          mods.abbreviate
          ? parse_month_abbr(s, parts.ymd_date.month)
          : parse_month_name(s, parts.ymd_date.month));
        break;

      case 'b':
        TRY(parse_month_abbr(s, parts.ymd_date.month));
        break;

      case 'D':
        TRY(parse_iso_date(s, parts.ymd_date, mods.abbreviate));
        break;

      case 'd':
        TRY(parse_day(s, parts.ymd_date.day));
        break;

      case 'G':
        TRY(parse_year(s, parts.week_date.week_year));
        break;

      case 'g':
        TRY(parse_two_digit_year(s, parts.week_date.week_year));
        break;

      case 'j':
        TRY(parse_ordinal(s, parts.ordinal_date.ordinal));
        break;

      case 'm':
        TRY(parse_month(s, parts.ymd_date.month));
        break;

      case 'u':
        TRY(parse_weekday_iso(s, parts.week_date.weekday));
        break;

      case 'V':
        TRY(parse_week(s, parts.week_date.week));
        break;

      case 'w':
        TRY(parse_weekday_unix(s, parts.week_date.weekday));
        break;

      case 'Y': 
        TRY(parse_year(s, parts.ymd_date.year)); 
        parts.ordinal_date.year = parts.ymd_date.year;
        break;

      case 'y':
        TRY(parse_two_digit_year(s, parts.ymd_date.year));
        break;

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


inline void
adjust(
  ParseExtra const& extra,
  HmsDaytime& hms)
{
  if (   hms.hour == HOUR_INVALID 
      && extra.hour_12 != HOUR_INVALID 
      && extra.am_pm != -1)
    // Replace hour with 12 hour and AM/PM indicator.
    hms.hour = 
        (extra.hour_12 == 12 ? 0 : extra.hour_12) 
      + (extra.am_pm == 1 ? 12 : 0);

  if (extra.usec != -1 && hms.second != SECOND_INVALID)
    // Replace fractional seconds with scaled usec.
    hms.second = int(hms.second) + extra.usec * 1e-6;
}


bool parse_daytime_parts(
  char const*& p,
  char const*& s,
  HmsDaytime& hms)
{
  // Second may be omitted.
  hms.second = 0;

  ParseExtra extra;

  while (true)
    if (*p == 0 && *s == 0) {
      // Completed successfully.
      adjust(extra, hms);
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
      case 'C': TRY(parse_iso_daytime(s, hms)); break;
      case 'f': TRY(parse_usec(s, extra.usec)); break;
      case 'H': TRY(parse_hour(s, hms.hour)); break;
      case 'I': TRY(parse_hour12(s, extra.hour_12)); break;
      case 'M': TRY(parse_minute(s, hms.minute)); break;
      case 'p': TRY(parse_am_pm(s, extra.am_pm)); break;
      case 'S': TRY(parse_second(s, hms.second)); break;

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

bool
parse_iso_time(
  char const*& s,
  YmdDate& date,
  HmsDaytime& hms,
  TimeZoneOffset& tz_offset,
  bool const compact)
{
  TRY(parse_iso_date(s, date, compact));
  if (*s == 'T' || *s == 't')
    ++s;
  else
    return false;
  TRY(parse_iso_daytime(s, hms, compact));
  // FIXME: Accept only 'Z' for UTC, not any other military time zone letter.
  if (*s == 'Z') {
    tz_offset = 0;
    ++s;
  }
  else
    TRY(parse_tz_offset(s, tz_offset));
  return true;
}


bool
parse_time_parts(
  char const*& p,
  char const*& s,
  FullDate& date,
  HmsDaytime& hms,
  TimeZoneInfo& tz)
{
  // Second may be omitted.
  hms.second = 0;

  daytime::ParseExtra extra;

  while (true)
    if (*p == 0 && *s == 0) {
      // Completed successfully.
      daytime::adjust(extra, hms);
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
      case 'A': TRY(parse_weekday_name(s, date.week_date.weekday)); break;
      case 'a': TRY(parse_weekday_abbr(s, date.week_date.weekday)); break;
      case 'B': TRY(parse_month_name(s, date.ymd_date.month)); break;
      case 'b': TRY(parse_month_abbr(s, date.ymd_date.month)); break;
      case 'D': TRY(parse_iso_date(s, date.ymd_date)); break;
      case 'd': TRY(parse_day(s, date.ymd_date.day)); break;
      case 'G': TRY(parse_year(s, date.week_date.week_year)); break;
      case 'g': TRY(parse_two_digit_year(s, date.week_date.week_year)); break;
      case 'j': TRY(parse_ordinal(s, date.ordinal_date.ordinal)); break;
      case 'm': TRY(parse_month(s, date.ymd_date.month)); break;
      case 'u': TRY(parse_weekday_iso(s, date.week_date.weekday)); break;
      case 'V': TRY(parse_week(s, date.week_date.week)); break;
      case 'w': TRY(parse_weekday_unix(s, date.week_date.weekday)); break;
      case 'Y': TRY(parse_year(s, date.ymd_date.year)); 
                date.ordinal_date.year = date.ymd_date.year; break;
      case 'y': TRY(parse_two_digit_year(s, date.ymd_date.year)); break;

      case 'C': TRY(parse_iso_daytime(s, hms)); break;
      case 'f': TRY(parse_usec(s, extra.usec)); break;
      case 'H': TRY(parse_hour(s, hms.hour)); break;
      case 'I': TRY(parse_hour12(s, extra.hour_12)); break;
      case 'M': TRY(parse_minute(s, hms.minute)); break;
      case 'p': TRY(parse_am_pm(s, extra.am_pm)); break;
      case 'S': TRY(parse_second(s, hms.second)); break;

      case 'E': TRY(parse_tz_offset(s, tz.offset)); break;
      case 'e': TRY(parse_tz_offset_letter(s, tz.offset)); break;
      case 'i': 
      case 'T': TRY(parse_iso_time(s, date.ymd_date, hms, tz.offset)); break;
      case 'o': TRY(parse_tz_offset_secs(s, tz.offset)); break;
      case 'Z': TRY(parse_tz_name(s, tz.name)); break;
      case 'z': TRY(parse_tz_offset(s, tz.offset, false)); break;

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


}  // namespace time

//------------------------------------------------------------------------------

}  // namespace ora

