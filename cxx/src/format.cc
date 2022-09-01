#include <algorithm>
#include <cassert>
#include <cstring>
#include <iomanip>
#include <iostream>

#include "ora/lib/exc.hh"
#include "ora/lib/string_builder.hh"
#include "ora/format.hh"

namespace ora {

using namespace ora::lib;
using ora::_impl::Modifiers;

using std::string;

//------------------------------------------------------------------------------
// Implementation helpers
//------------------------------------------------------------------------------

namespace {

bool
parse_modifiers(
  string const& pattern,
  size_t& pos,
  Modifiers& mods)
{
  switch (pattern[pos]) {
  case '.':
    if (mods.decimal)
      // Already saw a decimal point in this escape.
      throw ValueError("second decimal point in escape");
    else {
      mods.decimal = true;
      pos++;
    }
    break;

  case '0': case '1': case '2': case '3': case '4':
  case '5': case '6': case '7': case '8': case '9':
    {
      size_t const end = pattern.find_first_not_of("0123456789", pos);
      int const value = atoi(pattern.substr(pos, end).c_str());
      if (mods.decimal)
        mods.precision = value;
      else
        mods.width = value;
      pos = end;
    }
    break;

  case '#':
    mods.pad = pattern[++pos];
    if (pos++ == pattern.length())
      throw ValueError("unterminated escape in pattern");
    break;

  case '^':
  case '_':
    mods.str_case = pattern[pos];
    pos++;
    break;

  case '~':
    mods.abbreviate = true;
    pos++;
    break;

  default:
    // Did not match anything.
    return false;

  }

  // Matched something.
  return true;
}


void
format_string(
  StringBuilder& sb,
  Modifiers const& mods,
  std::string const& str)
{
  int const pad_length = mods.width - str.length();
  if (pad_length > 0)
    sb.pad(pad_length, mods.get_pad(' '));

  if (mods.str_case == '^' || mods.str_case == '_') {
    std::string formatted = str;
    std::transform(begin(formatted), end(formatted), begin(formatted), mods.str_case == '^' ? toupper : tolower);
    sb << formatted;
  }
  else
    sb << str;
}


bool
format_date(
  string const& pattern,
  size_t& pos,
  StringBuilder& sb,
  Modifiers const& mods,
  FullDate const& date)
{
  switch (pattern[pos]) {
  case 'A':
    format_string(
      sb, mods,
      mods.abbreviate ? get_weekday_abbr(date.week_date.weekday)
        : get_weekday_name(date.week_date.weekday));
    break;

  case 'a':
    format_string(sb, mods, get_weekday_abbr(date.week_date.weekday));
    break;

  case 'B':
    format_string(
      sb, mods,
      mods.abbreviate ? get_month_abbr(date.ymd_date.month)
        : get_month_name(date.ymd_date.month));
    break;

  case 'b':
    format_string(sb, mods, get_month_abbr(date.ymd_date.month));
    break;

  case 'D':
    sb.format(date.ymd_date.year, 4, '0');
    if (!mods.abbreviate)
      sb << '-';
    sb.format(date.ymd_date.month, 2, '0');
    if (!mods.abbreviate)
      sb << '-';
    sb.format(date.ymd_date.day, 2, '0');
    break;

  case 'd':
    sb.format(date.ymd_date.day, mods.get_width(2), mods.get_pad('0'));
    break;

  case 'G':
    sb.format(
      date.week_date.week_year, mods.get_width(4), mods.get_pad('0'));
    break;

  case 'g':
    sb.format(
      date.week_date.week_year % 100, mods.get_width(2), mods.get_pad('0'));
    break;

  case 'j':
    sb.format(
      date.ordinal_date.ordinal, mods.get_width(3), mods.get_pad('0'));
    break;

  case 'm':
    sb.format(
      date.ymd_date.month, mods.get_width(2), mods.get_pad('0'));
    break;

  case 'u':
    sb.format(
      weekday::ENCODING_ISO::encode(date.week_date.weekday),
      mods.get_width(1), mods.get_pad('0'));
    break;

  case 'V':
    sb.format(
      date.week_date.week, mods.get_width(2), mods.get_pad('0'));
    break;

  case 'w':
    sb.format(
      weekday::ENCODING_CLIB::encode(date.week_date.weekday),
      mods.get_width(1), mods.get_pad('0'));
    break;

  case 'y':
    sb.format(
      date.ymd_date.year % 100, mods.get_width(2), mods.get_pad('0'));
    break;

  case 'Y':
    sb.format(
      date.ymd_date.year, mods.get_width(4), mods.get_pad('0'));
    break;

  default:
    // Did not match anything.
    return false;

  }

  // Matched an output character.
  pos++;
  return true;
}


bool
format_daytime(
  string const& pattern,
  size_t& pos,
  StringBuilder& sb,
  Modifiers const& mods,
  HmsDaytime const& daytime)
{
  switch (pattern[pos]) {
  case 'C':
    daytime::format_iso_daytime(sb, daytime, mods.precision, mods.abbreviate);
    break;

  case 'f':
    {
      unsigned const usec = fmod(daytime.second, 1) * 1e6;
      sb.format(usec, 6, mods.get_pad('0'));
    }
    break;

  case 'H':
    sb.format(daytime.hour, mods.get_width(2), mods.get_pad('0'));
    break;

  case 'I':
    {
      unsigned const hour = daytime.hour % 12;
      sb.format(hour == 0 ? 12 : hour, mods.get_width(2), mods.get_pad('0'));
    }
    break;

  case 'M':
    sb.format(daytime.minute, mods.get_width(2), mods.get_pad('0'));
    break;

  case 'p':
    format_string(sb, mods, daytime.hour < 12 ? "AM" : "PM");
    break;

  case 'S':
    format_second(
      sb, daytime.second, mods.precision, mods.get_width(2), mods.get_pad('0'));
    break;

  default:
    // Did not match anything.
    return false;

  }

  // Matched an output character.
  pos++;
  return true;
}


bool
format_time_zone(
  string const& pattern,
  size_t& pos,
  StringBuilder& sb,
  Modifiers const& mods,
  TimeZoneParts const& time_zone)
{
  switch (pattern[pos]) {
  case 'E':
  case 'z':
    format_iso_offset(sb, time_zone, pattern[pos] == 'E', mods.get_width(2));
    break;

  case 'e':
    sb << get_time_zone_offset_letter(time_zone.offset);
    break;

  case 'o':
    sb << (time_zone.offset < 0 ? '-' : '+');
    sb.format(std::abs(time_zone.offset), mods.get_width(5), mods.get_pad('0'));
    break;

  case 'Z':
    // FIXME: Time zone full name.
    if (mods.abbreviate)
      sb << time_zone.abbreviation;
    else
      throw TimeFormatError("not implemented: time zone full name");
    break;

  default:
    // Did not match anything.
    return false;

  }

  // Matched an output character.
  pos++;
  return true;
}


bool
format_time(
  string const& pattern,
  size_t& pos,
  StringBuilder& sb,
  Modifiers const& mods,
  FullDate const& date,
  HmsDaytime const& daytime,
  TimeZoneParts const& time_zone)
{
  switch (pattern[pos]) {
  case 'c':
    // FIXME: Locale.
    throw TimeFormatError("not implemented: %c");
    break;

  case 'i':
  case 'T':
    time::format_iso_time(
      sb, date.ymd_date, daytime, time_zone, mods.precision, mods.abbreviate,
      mods.str_case != '_', pattern[pos] == 'T');
    break;

  default:
    // Did not match anything.
    return false;

  }

  // Matched an output character.
  pos++;
  return true;
}


}  // anonymous namespace


//------------------------------------------------------------------------------

namespace _impl {

inline size_t
find_next_escape(
  std::string const& pattern,
  size_t pos,
  StringBuilder& sb)
{
  while (true) {
    size_t const next = pattern.find('%', pos);
    if (next == std::string::npos) {
      // No next escape.  Copy the rest of the pattern, and we're done.
      sb << pattern.substr(pos);
      return std::string::npos;
    }
    else if (next > pos)
      // Copy from the pattern until the next escape.
      sb << pattern.substr(pos, next - pos);
    // Skip over the escape character.
    pos = next + 1;
    // Literal %?
    if (pos == pattern.length())
      throw ValueError("unterminated escape in pattern");
    else if (pattern[pos] == '%') {
      sb << '%';
      pos++;
      continue;
    }
    else
      return pos;
  }
}


std::string
Format::format(
  Datenum datenum)
  const
{
  auto const date = datenum_to_full_date(datenum);

  StringBuilder sb;
  for (size_t pos = find_next_escape(pattern_, 0, sb);
       pos != std::string::npos;
       pos = find_next_escape(pattern_, pos, sb)) {
    // Set up state for the escape sequence.
    Modifiers mods;
    // Scan characters in the escape sequence.
    while (parse_modifiers(pattern_, pos, mods))
      ;
    if (!format_date(pattern_, pos, sb, mods, date))
      throw TimeFormatError(
        std::string("unknown date escape '") + pattern_[pos] + "'");
  }
  return sb.str();
}


std::string
Format::format(
  HmsDaytime const& hms)
  const
{
  StringBuilder sb;
  for (size_t pos = find_next_escape(pattern_, 0, sb);
       pos != std::string::npos;
       pos = find_next_escape(pattern_, pos, sb)) {
    // Set up state for the escape sequence.
    Modifiers mods;
    // Scan characters in the escape sequence.
    while (parse_modifiers(pattern_, pos, mods))
      ;
    if (!format_daytime(pattern_, pos, sb, mods, hms))
      throw TimeFormatError(
        std::string("unknown daytime escape '") + pattern_[pos] + "'");
  }
  return sb.str();
}


std::string
Format::format(
  LocalDatenumDaytick const& ldd)
  const
{
  auto const date = datenum_to_full_date(ldd.datenum);
  auto const hms = daytick_to_hms(ldd.daytick);

  StringBuilder sb;
  for (size_t pos = find_next_escape(pattern_, 0, sb);
       pos != std::string::npos;
       pos = find_next_escape(pattern_, pos, sb)) {
    // Set up state for the escape sequence.
    Modifiers mods;
    // Scan characters in the escape sequence.
    while (parse_modifiers(pattern_, pos, mods))
      ;
    if (   !format_time(pattern_, pos, sb, mods, date, hms, ldd.time_zone)
        && !format_date(pattern_, pos, sb, mods, date)
        && !format_daytime(pattern_, pos, sb, mods, hms)
        && !format_time_zone(pattern_, pos, sb, mods, ldd.time_zone))
      throw TimeFormatError(
        std::string("unknown time escape '") + pattern_[pos] + "'");
  }
  return sb.str();
}


}  // namespace _impl

//------------------------------------------------------------------------------
// Class TimeFormat
//------------------------------------------------------------------------------

namespace time {

TimeFormat const TimeFormat::DEFAULT("%i", "INVALID", "MISSING");
TimeFormat const TimeFormat::ISO_LOCAL_BASIC            = "%Y%m%dT%H%M%S";
TimeFormat const TimeFormat::ISO_LOCAL_EXTENDED         = "%Y-%m-%dT%H:%M:%S";
TimeFormat const TimeFormat::ISO_ZONE_LETTER_BASIC      = "%Y%m%dT%H%M%S%e";
TimeFormat const TimeFormat::ISO_ZONE_LETTER_EXTENDED   = "%Y-%m-%dT%H:%M:%S%e";
TimeFormat const TimeFormat::ISO_ZONE_BASIC             = "%~i";
TimeFormat const TimeFormat::ISO_ZONE_EXTENDED          = "%i";

}  // namespace time

//------------------------------------------------------------------------------
// Class DateFormat
//------------------------------------------------------------------------------

namespace date {

DateFormat const DateFormat::DEFAULT("%Y-%m-%d", "INVALID", "MISSING");
DateFormat const DateFormat::ISO_CALENDAR_BASIC    = "%Y%m%d";
DateFormat const DateFormat::ISO_CALENDAR_EXTENDED = "%Y-%m-%d";
DateFormat const DateFormat::ISO_ORDINAL_BASIC     = "%Y%j";
DateFormat const DateFormat::ISO_ORDINAL_EXTENDED  = "%Y-%j";
DateFormat const DateFormat::ISO_WEEK_BASIC        = "%GW%V%w";
DateFormat const DateFormat::ISO_WEEK_EXTENDED     = "%G-W%V-%w";

}  // namespace date

//------------------------------------------------------------------------------
// Class DaytimeFormat
//------------------------------------------------------------------------------

namespace daytime {

DaytimeFormat const DaytimeFormat::DEFAULT("%H:%M:%.15S", "INVALID", "DEFAULT");
DaytimeFormat const DaytimeFormat::ISO_BASIC("%H%M%S", "INVALD", "MISSNG");
DaytimeFormat const DaytimeFormat::ISO_EXTENDED         = "%H:%M:%S";
DaytimeFormat const DaytimeFormat::ISO_BASIC_MSEC       = "%H%M%.3S";
DaytimeFormat const DaytimeFormat::ISO_EXTENDED_MSEC    = "%H:%M:%.3S";
DaytimeFormat const DaytimeFormat::ISO_BASIC_USEC       = "%H%M%.6S";
DaytimeFormat const DaytimeFormat::ISO_EXTENDED_USEC    = "%H:%M:%.6S";
DaytimeFormat const DaytimeFormat::ISO_BASIC_NSEC       = "%H%M%.9S";
DaytimeFormat const DaytimeFormat::ISO_EXTENDED_NSEC    = "%H:%M:%.9S";

}  // namespace daytime

//------------------------------------------------------------------------------

// FIXME: Internationalize.

namespace {

string const
month_names[] = {
  "January",
  "February",
  "March",
  "April",
  "May",
  "June",
  "July",
  "August",
  "September",
  "October",
  "November",
  "December"
};


string const
month_abbrs[] = {
  "Jan", "Feb", "Mar", "Apr", "May", "Jun",
  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
};


// Letters representing military time zone offsets, starting with UTC-12:00 and
// proceeding in one-hour increments through UTC+12:00.
char const
time_zone_offset_letters[25] = {
  'Y', 'X', 'W', 'V', 'U', 'T', 'S', 'R', 'Q', 'P', 'O', 'N',
  'Z',
  'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M',
};


// Inverse of the above, for 'A' through 'Z'.
TimeZoneOffset const
time_zone_letter_offsets[26] = {
   1 * (int) SECS_PER_HOUR,    // A
   2 * (int) SECS_PER_HOUR,    // B
   3 * (int) SECS_PER_HOUR,    // C
   4 * (int) SECS_PER_HOUR,    // D
   5 * (int) SECS_PER_HOUR,    // E
   6 * (int) SECS_PER_HOUR,    // F
   7 * (int) SECS_PER_HOUR,    // G
   8 * (int) SECS_PER_HOUR,    // H
   9 * (int) SECS_PER_HOUR,    // I
  TIME_ZONE_OFFSET_INVALID,    // J
  10 * (int) SECS_PER_HOUR,    // K
  11 * (int) SECS_PER_HOUR,    // L
  12 * (int) SECS_PER_HOUR,    // M
  -1 * (int) SECS_PER_HOUR,    // N
  -2 * (int) SECS_PER_HOUR,    // O
  -3 * (int) SECS_PER_HOUR,    // P
  -4 * (int) SECS_PER_HOUR,    // Q
  -5 * (int) SECS_PER_HOUR,    // R
  -6 * (int) SECS_PER_HOUR,    // S
  -7 * (int) SECS_PER_HOUR,    // T
  -8 * (int) SECS_PER_HOUR,    // U
  -9 * (int) SECS_PER_HOUR,    // V
 -10 * (int) SECS_PER_HOUR,    // W
 -11 * (int) SECS_PER_HOUR,    // X
 -12 * (int) SECS_PER_HOUR,    // Y
   0                ,          // Z
};

// The letter to use for a time zone offset that has no military / nautical
// correspondence, because it is not a round hour offset from UTC.
char const
time_zone_offset_letter_missing
  = '?';


string const
weekday_names[] = {
  "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday" };


string const
weekday_abbrs[] = { "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun" };


}  // anonymous


string const&
get_month_name(
  Month const month)
{
  if (! month_is_valid(month))
    throw ValueError("bad month");
  return month_names[(int) month - 1];
}


Month
parse_month_name(
  string const& str)
{
  for (Month month = MONTH_MIN; month < MONTH_END; ++month)
    if (month_names[(int) month - 1] == str)
      return month;
  throw ValueError(string("bad month name: ") + str);
}


bool
parse_month_name(
  char const*& p,
  Month& month)
{
  for (Month m = MONTH_MIN; m < MONTH_END; ++m) {
    auto const& name = month_names[(int) m - 1];
    if (strncmp(name.c_str(), p, name.length()) == 0) {
      p += name.length();
      month = m;
      return true;
    }
  }
  return false;
}


string const&
get_month_abbr(
  Month const month)
{
  if (! month_is_valid(month))
    throw ValueError("bad month");
  return month_abbrs[(int) month - 1];
}


Month
parse_month_abbr(
  string const& str)
{
  for (Month month = MONTH_MIN; month < MONTH_END; ++month)
    if (month_abbrs[(int) month - 1] == str)
      return month;
  throw ValueError(string("bad month abbr: ") + str);
}


bool
parse_month_abbr(
  char const*& p,
  Month& month)
{
  for (Month m = MONTH_MIN; m < MONTH_END; ++m) {
    auto const& abbr = month_abbrs[(int) m - 1];
    if (strncmp(abbr.c_str(), p, abbr.length()) == 0) {
      p += abbr.length();
      month = m;
      return true;
    }
  }
  return false;
}


char
get_time_zone_offset_letter(
  TimeZoneOffset const offset)
{
  // Fast path.
  if (offset == 0)
    return 'Z';

  auto hours = std::div(offset, SECS_PER_HOUR);
  if (hours.rem == 0) {
    assert(0 <= hours.quot + 12 && hours.quot + 12 <= 24);
    return time_zone_offset_letters[hours.quot + 12];
  }
  else
    return time_zone_offset_letter_missing;
}


string const&
get_weekday_name(
  Weekday weekday)
{
  if (! weekday_is_valid(weekday))
    throw ValueError("bad weekday");
  return weekday_names[(int) weekday];
}


// FIXME: Move parsing functions to parse.cc.

TimeZoneOffset
parse_time_zone_offset_letter(
  char const letter)
{
  return
      'A' <= letter && letter <= 'Z'
    ? time_zone_letter_offsets[letter - 'A']
    : TIME_ZONE_OFFSET_INVALID;
}


bool
parse_weekday_name(
  char const*& p,
  Weekday& weekday)
{
  for (Weekday w = 0; w < 7; ++w) {
    auto const& name = weekday_names[w];
    if (strncasecmp(name.c_str(), p, name.length()) == 0) {
      p += name.length();
      weekday = w;
      return true;
    }
  }
  return false;
}


Weekday
parse_weekday_name(
  string const& str)
{
  Weekday weekday;
  char const* p = str.c_str();
  if (parse_weekday_name(p, weekday))
    return weekday;
  else
    throw ValueError(string("bad weekday name: ") + str);
}


string const&
get_weekday_abbr(
  Weekday weekday)
{
  if (! weekday_is_valid(weekday))
    throw ValueError("bad weekday");
  return weekday_abbrs[(int) weekday];
}


bool
parse_weekday_abbr(
  char const*& p,
  Weekday& weekday)
{
  for (Weekday w = 0; w < 7; ++w) {
    auto const& abbr = weekday_abbrs[w];
    if (strncasecmp(abbr.c_str(), p, abbr.length()) == 0) {
      p += abbr.length();
      weekday = w;
      return true;
    }
  }
  return false;
}


Weekday
parse_weekday_abbr(
  string const& str)
{
  Weekday weekday;
  char const* p = str.c_str();
  if (parse_weekday_abbr(p, weekday))
    return weekday;
  else
    throw ValueError(string("bad weekday abbr: ") + str);
}


//------------------------------------------------------------------------------

}  // namespace ora
