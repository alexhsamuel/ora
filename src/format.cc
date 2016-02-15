#include <algorithm>
#include <cassert>
#include <cstring>
#include <iomanip>
#include <iostream>

#include "cron/format.hh"
#include "exc.hh"
#include "string_builder.hh"

using std::string;

namespace alxs {
namespace cron {

//------------------------------------------------------------------------------
// Implementation helpers
//------------------------------------------------------------------------------

namespace {

/**
 * Helper class to hold modifier state in an escape sequence.
 */
struct Modifiers
{
  /**
   * Returns the numeric width, or a default value if it's not set.
   */
  int get_width(int def) const { return width == -1 ? def : width; }
  
  /**
   * Returns the pad character, or a default value if it's not set.
   */
  char get_pad(char def) const { return pad == 0 ? def : pad; }

  int width = -1;
  int precision = -1;
  char pad = 0;
  char str_case = 0;
  bool abbreviate = false;
  bool decimal = false;

};


inline bool
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

  case 'E':
    // FIXME: IMPLEMENT: Locale's alternative representation.
    throw TimeFormatError("not implemented: E");
    break;

  case 'O':
    // FIXME: IMPLEMENT: Locale's alternative numerical representation
    throw TimeFormatError("not implemented: O");
    break;

  default:
    // Did not match anything.
    return false;

  }

  // Matched something.
  return true;
}


inline void
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


inline bool
format_date(
  string const& pattern,
  size_t& pos,
  StringBuilder& sb,
  Modifiers const& mods,
  Parts const& parts)
{
  switch (pattern[pos]) {
  case 'b':
    format_string(sb, mods, mods.abbreviate ? get_month_abbr(parts.date_.month) : get_month_name(parts.date_.month));
    break;

  case 'd':
    sb.format(parts.date_.day + 1, mods.get_width(2), mods.get_pad('0'));
    break;

  case 'D':
    // FIXME: Locale.
    throw TimeFormatError("not implemented: %D");
    break;

  case 'g':
    sb.format(parts.week_date_.week_year % 100, mods.get_width(2), mods.get_pad('0'));
    break;

  case 'G':
    sb.format(parts.week_date_.week_year, mods.get_width(4), mods.get_pad('0'));
    break;

  case 'j':
    sb.format(parts.ordinal_date_.ordinal + 1, mods.get_width(3), mods.get_pad('0'));
    break;

  case 'm':
    sb.format(parts.date_.month + 1, mods.get_width(2), mods.get_pad('0'));
    break;

  case 'V':
    sb.format(parts.week_date_.week + 1, mods.get_width(2), mods.get_pad('0'));
    break;

  case 'w':
    // FIXME: Generalize?
    sb.format((parts.week_date_.weekday + (7 - SUNDAY)) % 7, mods.get_width(1), mods.get_pad('0'));
    break;

  case 'W':
    format_string(sb, mods, mods.abbreviate ? get_weekday_abbr(parts.week_date_.weekday) : get_weekday_name(parts.week_date_.weekday));
    break;

  case 'y':
    sb.format(parts.date_.year % 100, mods.get_width(2), mods.get_pad('0'));
    break;

  case 'Y':
    sb.format(parts.date_.year, mods.get_width(4), mods.get_pad('0'));
    break;

  default:
    // Did not match anything.
    return false;

  }

  // Matched an output character.
  pos++;
  return true;
}


inline bool
format_daytime(
  string const& pattern,
  size_t& pos,
  StringBuilder& sb,
  Modifiers const& mods,
  DaytimeParts const& parts)
{
  switch (pattern[pos]) {
  case 'h':
    {
      unsigned const hour = parts.hour % 12;
      sb.format(hour == 0 ? 12 : hour, mods.get_width(2), mods.get_pad('0'));
    }
    break;

  case 'H':
    sb.format(parts.hour, mods.get_width(2), mods.get_pad('0'));
    break;

  case 'k':
    {
      unsigned const msec = (parts.second - (unsigned) parts.second) * 1e+3;
      sb.format(msec, mods.get_width(3), mods.get_pad('0'));
    }
    break;

  case 'K':
    {
      unsigned const usec = (unsigned) ((parts.second - (unsigned) parts.second) * 1e+6) % 1000;
      sb.format(usec, mods.get_width(3), mods.get_pad('0'));
    }
    break;

  case 'l':
    {
      unsigned const nsec = (unsigned) ((parts.second - (unsigned) parts.second) * 1e+9) % 1000;
      sb.format(nsec, mods.get_width(3), mods.get_pad('0'));
    }
    break;

  case 'L':
    {
      unsigned const psec = (unsigned) ((parts.second - (unsigned) parts.second) * 1e+12) % 1000;
      sb.format(psec, mods.get_width(3), mods.get_pad('0'));
    }
    break;

  case 'M':
    sb.format(parts.minute, mods.get_width(2), mods.get_pad('0'));
    break;

  case 'p':
    format_string(sb, mods, parts.hour < 12 ? "AM" : "PM");
    break;

  case 'S':
    {
      unsigned long const prec = std::max(0, mods.precision);
      unsigned long long const digits = parts.second * pow10(prec) + 0.5;
      // Integer part.
      sb.format(digits / pow10(prec), mods.get_width(2), mods.get_pad('0'));
      if (mods.precision >= 0) {
        sb << '.';
        // Fractional part.
        if (mods.precision > 0) 
          sb.format(digits % pow10(prec), prec, '0');
      }
    }
    break;

  case 'T':
    // FIXME: Locale.
    throw TimeFormatError("not implemented: %T");
    break;

  default:
    // Did not match anything.
    return false;

  }

  // Matched an output character.
  pos++;
  return true;
}


inline bool
format_time_zone(
  string const& pattern,
  size_t& pos,
  StringBuilder& sb,
  Modifiers const& mods,
  TimeZoneParts const& parts)
{
  switch (pattern[pos]) {
  case 'o':
    sb << (parts.offset < 0 ? '-' : '+');
    sb.format(std::abs(parts.offset), mods.get_width(5), mods.get_pad('0'));
    break;

  case 'q':
    {
      unsigned const offset_min = std::abs(parts.offset) % SECS_PER_HOUR / SECS_PER_MIN;
      sb.format(offset_min, mods.get_width(2), mods.get_pad('0'));
    }
    break;

  case 'Q':
    {
      unsigned const offset_hour = std::abs(parts.offset) / SECS_PER_HOUR;
      sb.format(offset_hour, mods.get_width(2),mods.get_pad('0'));
    }
    break;

  case 'U':
    sb << (parts.offset < 0 ? '-' : '+');
    break;

  case 'Z':
    // FIXME: Time zone full name.
    if (mods.abbreviate)
      sb << parts.abbreviation;
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


inline bool
format_time(
  string const& pattern,
  size_t& pos,
  StringBuilder& /* sb */,
  Modifiers const& /* mods */,
  Parts const& /* parts */)
{
  switch (pattern[pos]) {
  case 'c':
    // FIXME: Locale.
    throw TimeFormatError("not implemented: %c");
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

void 
Format::format(
  StringBuilder& sb,
  Parts const& parts)
  const
{
  size_t pos = 0;
  while (true) {
    // Find the next escape character.
    size_t const next = pattern_.find('%', pos);
    if (next == std::string::npos) {
      // No next escape.  Copy the rest of the pattern, and we're done.
      sb << pattern_.substr(pos);
      break;
    }
    else if (next > pos)
      // Copy from the pattern until the next escape.
      sb << pattern_.substr(pos, next - pos);
    // Skip over the escape character.
    pos = next + 1;

    // Set up state for the escape sequence.
    Modifiers mods;

    // Scan characters in the escape sequence.
    for (bool done = false; ! done; ) {
      if (pos == pattern_.length())
        throw ValueError("unterminated escape in pattern");

      // Literal '%' escape.
      if (pattern_[pos] == '%') {
        sb << '%';
        pos++;
        break;
      }

      // Handle modifiers.
      if (parse_modifiers(pattern_, pos, mods))
        continue;

      // Handle escape codes for date components.
      if (   format_date        (pattern_, pos, sb, mods, parts)
          || format_daytime     (pattern_, pos, sb, mods, parts.daytime_)
          || format_time_zone   (pattern_, pos, sb, mods, parts.time_zone_)
          || format_time        (pattern_, pos, sb, mods, parts))
        break;

      // If we made it this far, it's not a valid character.
      throw TimeFormatError(
        std::string("unknown escape '") + pattern_[pos] + "'");
    }
  }
}


//------------------------------------------------------------------------------
// Class TimeFormat
//------------------------------------------------------------------------------

TimeFormat const TimeFormat::ISO_LOCAL_BASIC    = "%Y%m%dT%H%M%S";
TimeFormat const TimeFormat::ISO_LOCAL_EXTENDED = "%Y-%m-%dT%H:%M:%S";
TimeFormat const TimeFormat::ISO_UTC_BASIC      = "%Y%m%dT%H%M%SZ";
TimeFormat const TimeFormat::ISO_UTC_EXTENDED   = "%Y-%m-%dT%H:%M:%SZ";
TimeFormat const TimeFormat::ISO_ZONE_BASIC     = "%Y%m%dT%H%M%S%U%Q%q";
TimeFormat const TimeFormat::ISO_ZONE_EXTENDED  = "%Y-%m-%dT%H:%M:%S%U%Q:%q";

//------------------------------------------------------------------------------
// Class DateFormat
//------------------------------------------------------------------------------

DateFormat const DateFormat::ISO_CALENDAR_BASIC    = "%Y%m%d";
DateFormat const DateFormat::ISO_CALENDAR_EXTENDED = "%Y-%m-%d";
DateFormat const DateFormat::ISO_ORDINAL_BASIC     = "%Y%j";
DateFormat const DateFormat::ISO_ORDINAL_EXTENDED  = "%Y-%j";
DateFormat const DateFormat::ISO_WEEK_BASIC        = "%GW%V%^w";
DateFormat const DateFormat::ISO_WEEK_EXTENDED     = "%G-W%V-%^w";

//------------------------------------------------------------------------------
// Class DaytimeFormat
//------------------------------------------------------------------------------

DaytimeFormat const DaytimeFormat::ISO_BASIC("%H%M%S", "INVALD", "MISSNG");
DaytimeFormat const DaytimeFormat::ISO_EXTENDED         = "%H:%M:%S";
DaytimeFormat const DaytimeFormat::ISO_BASIC_MSEC       = "%H%M%.3S";
DaytimeFormat const DaytimeFormat::ISO_EXTENDED_MSEC    = "%H:%M:%.3S";
DaytimeFormat const DaytimeFormat::ISO_BASIC_USEC       = "%H%M%.6S";
DaytimeFormat const DaytimeFormat::ISO_EXTENDED_USEC    = "%H:%M:%.6S";
DaytimeFormat const DaytimeFormat::ISO_BASIC_NSEC       = "%H%M%.9S";
DaytimeFormat const DaytimeFormat::ISO_EXTENDED_NSEC    = "%H:%M:%.9S";

//------------------------------------------------------------------------------

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


string const
weekday_names[] = {
  "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday" };


string const
weekday_abbrs[] = { "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun" };


}  // anonymous


inline string const& 
get_month_name(
  Month month)
{
  if (! month_is_valid(month))
    throw ValueError("bad month");
  return month_names[(int) month];
}


inline Month 
parse_month_name(
  string const& str)
{
  for (Month month = 0; month < 12; ++month)
    if (month_names[month] == str)
      return month;
  throw ValueError(string("bad month name: ") + str);
}


inline string const& 
get_month_abbr(
  Month month)
{
  if (! month_is_valid(month))
    throw ValueError("bad month");
  return month_abbrs[(int) month];
}


inline Month 
parse_month_abbr(
  string const& str)
{
  for (Month month = 0; month < 12; ++month)
    if (month_abbrs[month] == str)
      return month;
  throw ValueError(string("bad month abbr: ") + str);
}


inline string const& 
get_weekday_name(
  Weekday weekday)
{
  if (! weekday_is_valid(weekday))
    throw ValueError("bad weekday");
  return weekday_names[(int) weekday];
}


inline Weekday 
parse_weekday_name(
  string const& str)
{
  for (Weekday weekday = 0; weekday < 7; ++weekday)
    if (weekday_names[weekday] == str)
      return weekday;
  throw ValueError(string("bad weekday name: ") + str);
}


inline string const& 
get_weekday_abbr(
  Weekday weekday)
{
  if (! weekday_is_valid(weekday))
    throw ValueError("bad weekday");
  return weekday_abbrs[(int) weekday];
}


Weekday 
parse_weekday_abbr(
  string const& str)
{
  for (Weekday weekday = 0; weekday < 7; ++weekday)
    if (weekday_abbrs[weekday] == str)
      return weekday;
  throw ValueError(string("bad weekday abbr: ") + str);
}


//------------------------------------------------------------------------------

}  // namespace cron
}  // namespace alxs

