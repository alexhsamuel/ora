#pragma once

#include <algorithm>
#include <iostream>
#include <sstream>
#include <stack>
#include <string>

#include "ora/lib/exc.hh"
#include "ora/lib/string.hh"
#include "ora/lib/string_builder.hh"
#include "ora/date_type.hh"
#include "ora/daytime_type.hh"
#include "ora/time_type.hh"
#include "ora/time_zone.hh"
#include "ora/types.hh"

namespace ora {

//------------------------------------------------------------------------------
// Functions
//------------------------------------------------------------------------------

std::string const& get_month_name(Month month);
Month parse_month_name(std::string const& str);
bool parse_month_name(char const*& p, Month& month);
std::string const& get_month_abbr(Month month);
Month parse_month_abbr(std::string const& str);
bool parse_month_abbr(char const*& p, Month& month);

/*
 * Returns the military / nautical time zone offset letter.
 *
 * Returns the one letter code of the [military time
 * zone](http://en.wikipedia.org/wiki/List_of_military_time_zones)
 * corresponding to an offset.  If there is no military time zone, returns an
 * unspecified nonalphabetic character.
 */
char get_time_zone_offset_letter(TimeZoneOffset);

/*
 * Returns the time zone offset represented by tbe military / time zone offset
 * letter.
 *
 * If the letter is invalid, returns TIME_ZONE_OFFSET_INVALID.
 */
TimeZoneOffset parse_time_zone_offset_letter(char letter);

std::string const& get_weekday_name(Weekday weekday);
Weekday parse_weekday_name(std::string const& str);
bool parse_weekday_name(char const*& p, Weekday& weekday);
std::string const& get_weekday_abbr(Weekday weekday);
Weekday parse_weekday_abbr(std::string const& str);
bool parse_weekday_abbr(char const*& p, Weekday& weekday);

//------------------------------------------------------------------------------

namespace {

inline void
format_second(
  StringBuilder& sb,
  Second const second,
  int const precision=-1,
  int const width=2,
  char const pad='0',
  bool const trim=false)
{
  // FIXME: Improve this logic.  See fixfmt, or convert to a power-of-10
  // denominator.
  unsigned const prec = std::max(0, precision);
  long long const prec10 = pow10(prec);
  auto const digits = std::div((long long) (second * prec10), prec10);
  // Integer part.
  sb.format(digits.quot, width, pad);
  if (precision >= 0) {
    sb << '.';
    // Fractional part.
    if (precision > 0)
      sb.format(digits.rem, prec, '0');
    if (trim) {
      sb.rstrip('0');
      sb.rstrip('.');
    }
  }
}


inline void
format_iso_offset(
  StringBuilder& sb,
  TimeZoneParts const& time_zone,
  bool const colon=true,
  int const width=2)
{
  sb << (time_zone.offset < 0 ? '-' : '+');
  auto const off = std::abs(time_zone.offset);
  auto const hr = off / SECS_PER_HOUR;
  auto const mn = off % SECS_PER_HOUR / SECS_PER_MIN;
  sb.format(hr, width, '0');
  if (colon)
    sb << ':';
  sb.format(mn, width, '0');
}


}  // anonymous namespace

//------------------------------------------------------------------------------

namespace _impl {

using namespace ora::lib;

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


class Format
{
public:

  Format(
    std::string const& pattern,
    std::string const& invalid="INVALID",
    std::string const& missing="MISSING")
  : pattern_(pattern),
    invalid_(invalid),
    missing_(missing)
  {
  }

  Format(
    char const* pattern)
  : pattern_(pattern),
    invalid_("INVALID"),
    missing_("MISSING")
  {
  }

  size_t
  get_width()
    const
  {
    set_up_width();
    return (size_t) width_;
  }

  std::string const& get_pattern() const { return pattern_; }
  std::string const& get_invalid() const { return invalid_; }
  std::string const& get_missing() const { return missing_; }

protected:

  std::string format(Datenum) const;
  std::string format(HmsDaytime const&) const;
  std::string format(LocalDatenumDaytick const&) const;

  std::string const&
  get_invalid_pad()
    const
  {
    set_up_width();
    return invalid_pad_;
  }

  std::string const&
  get_missing_pad()
    const
  {
    set_up_width();
    return missing_pad_;
  }

private:

  void
  set_up_width()
    const
  {
    if (width_ == -1) {
      // Find the length of a formatted string.
      static auto const MIN_TIME = LocalDatenumDaytick{
        DATENUM_MIN,
        DAYTICK_MIN,
        TimeZoneParts{0, "", false},
      };
      width_ = (int) format(MIN_TIME).length();

      // Truncate or pad the invalid and missing strings.
      invalid_pad_ = pad_trunc(invalid_, width_, ' ');
      missing_pad_ = pad_trunc(missing_, width_, ' ');
    }
  }

  std::string const pattern_;
  std::string const invalid_;
  std::string const missing_;

  mutable int width_ = -1;
  mutable std::string invalid_pad_;
  mutable std::string missing_pad_;

};


}  // namespace _impl

//------------------------------------------------------------------------------

namespace daytime {

inline void
format_iso_daytime(
  StringBuilder& sb,
  HmsDaytime const& daytime,
  int const precision,
  bool const compact=false,
  bool const trim=false)
{
  sb.format(daytime.hour, 2, '0');
  if (!compact)
    sb << ':';
  sb.format(daytime.minute, 2, '0');
  if (!compact)
    sb << ':';
  format_second(sb, daytime.second, precision, 2, '0', trim);
}


class DaytimeFormat
  : public _impl::Format
{
public:

  static DaytimeFormat const DEFAULT;
  static DaytimeFormat const ISO_BASIC;
  static DaytimeFormat const ISO_EXTENDED;
  static DaytimeFormat const ISO_BASIC_MSEC;
  static DaytimeFormat const ISO_EXTENDED_MSEC;
  static DaytimeFormat const ISO_BASIC_USEC;
  static DaytimeFormat const ISO_EXTENDED_USEC;
  static DaytimeFormat const ISO_BASIC_NSEC;
  static DaytimeFormat const ISO_EXTENDED_NSEC;

  using Format::Format;

  std::string
  operator()(
    HmsDaytime const& hms)
    const
  {
    return format(hms);
  }

  template<class TRAITS>
  std::string
  operator()(
    DaytimeTemplate<TRAITS> const daytime,
    bool const fixed=true)
    const
  {
    return
        daytime.is_invalid() ? (fixed ? get_invalid_pad() : get_invalid())
      : daytime.is_missing() ? (fixed ? get_missing_pad() : get_missing())
      : format(get_hms(daytime));
  }

};


template<class TRAITS>
inline std::string
to_string(
  DaytimeTemplate<TRAITS> const daytime)
{
  return DaytimeFormat::DEFAULT(daytime, false);
}


template<class TRAITS>
inline std::ostream&
operator<<(
  std::ostream& os,
  DaytimeTemplate<TRAITS> const daytime)
{
  os << to_string(daytime);
  return os;
}


}  // namespace daytime

//------------------------------------------------------------------------------

namespace time {

inline void
format_iso_time(
  StringBuilder& sb,
  YmdDate date,
  HmsDaytime daytime,
  TimeZoneParts const& time_zone,
  int const precision,
  bool const compact=false,
  bool const capital=true,
  bool const military=false,
  bool const trim=false,
  bool const round=false)
{
  sb.format(date.year, 4, '0');
  if (!compact)
    sb << '-';
  sb.format(date.month, 2, '0');
  if (!compact)
    sb << '-';
  sb.format(date.day, 2, '0');
  sb << (capital ? 'T' : 't');
  daytime::format_iso_daytime(sb, daytime, precision, compact, trim);
  if (military)
    sb << get_time_zone_offset_letter(time_zone.offset);
  else
    format_iso_offset(sb, time_zone, !compact);
}


class TimeFormat
  : public _impl::Format
{
public:

  static TimeFormat const DEFAULT;
  static TimeFormat const ISO_LOCAL_BASIC;
  static TimeFormat const ISO_LOCAL_EXTENDED;
  static TimeFormat const ISO_ZONE_LETTER_BASIC;
  static TimeFormat const ISO_ZONE_LETTER_EXTENDED;
  static TimeFormat const ISO_ZONE_BASIC;
  static TimeFormat const ISO_ZONE_EXTENDED;

  using Format::Format;

  std::string
  operator()(
    LocalDatenumDaytick const& ldd)
    const
  {
    return format(ldd);
  }

  /*
   * If `fixed` is true, the result is of fixed width.
   */
  template<class TIME>
  std::string
  operator()(
    TIME const time,
    TimeZone const& time_zone=*UTC,
    bool const fixed=true)
    const
  {
    if (time.is_invalid())
      return fixed ? get_invalid_pad() : get_invalid();
    else if (time.is_missing())
      return fixed ? get_missing_pad() : get_missing();
    else
      // FIXME: Don't use daytick, which may lose precision.
      return operator()(to_local_datenum_daytick(time, time_zone));
  }

  template<class TIME>
  std::string
  operator()(
    TIME const time,
    std::string const& tz_name,
    bool const fixed=true)
    const
  {
    return operator()(time, get_time_zone(tz_name), fixed);
  }

  template<class TIME>
  std::string operator()(
    TIME const time,
    _DisplayTimeZoneTag,
    bool const fixed=true)
    const
  {
    return operator()(time, *get_display_time_zone(), fixed);
  }

};


template<class TRAITS>
inline std::string
to_string(
  TimeType<TRAITS> const time,
  TimeZone const& time_zone=*UTC)
{
  return TimeFormat::DEFAULT(time, time_zone, false);
}


template<class TRAITS>
inline std::string
to_string(
  TimeType<TRAITS> const time,
  _DisplayTimeZoneTag)
{
  return to_string(time, *get_display_time_zone(), false);
}


template<class TRAITS>
inline std::ostream&
operator<<(
  std::ostream& os,
  TimeType<TRAITS> const time)
{
  os << TimeFormat::DEFAULT(time);
  return os;
}


class LocalTimeFormat
{
public:

  LocalTimeFormat(
    std::string const& pattern,
    TimeZone_ptr tz)
  : fmt_(pattern),
    tz_(tz)
  {
  }

  static LocalTimeFormat
  parse(
    std::string const& pattern)
  {
    if (pattern.length() == 0)
      // Empty pattern.
      return {TimeFormat::DEFAULT.get_pattern(), UTC};
    // Look for a time zone in the format pattern.
    auto const at = pattern.rfind('@');
    if (at == std::string::npos) {
      // No time zone given; use UTC.
      return {pattern, UTC};
    }
    else {
      auto const tz_name = pattern.substr(at + 1);
      TimeZone_ptr tz;
      if (tz_name == "" || tz_name == "display")
        tz = ora::get_display_time_zone();
      else if (tz_name == "UTC")
        tz = UTC;
      else if (tz_name == "system")
        try {
          tz = ora::get_system_time_zone();
        }
        catch (RuntimeError const&) {
          tz = UTC;
        }
      else
        tz = ora::get_time_zone(tz_name);
      if (at == 0)
        // Empty pattern part; use ISO format.
        return {TimeFormat::DEFAULT.get_pattern(), tz};
      else
        return {pattern.substr(0, at), tz};
   }
  }

  template<class TIME>
  std::string
  operator()(
    TIME const time)
    const
  {
    return fmt_(time, *tz_);
  }

private:

  TimeFormat fmt_;
  TimeZone_ptr tz_;

};


}  // namespace time

//------------------------------------------------------------------------------

namespace date {

class DateFormat
  : public _impl::Format
{
public:

  static DateFormat const DEFAULT;
  static DateFormat const ISO_CALENDAR_BASIC;
  static DateFormat const ISO_CALENDAR_EXTENDED;
  static DateFormat const ISO_ORDINAL_BASIC;
  static DateFormat const ISO_ORDINAL_EXTENDED;
  static DateFormat const ISO_WEEK_BASIC;
  static DateFormat const ISO_WEEK_EXTENDED;

  using Format::Format;

  template<class DATE>
  std::string
  operator()(
    DATE const date,
    bool const fixed=true)
    const
  {
    return
        date.is_invalid() ? (fixed ? get_invalid_pad() : get_invalid())
      : date.is_missing() ? (fixed ? get_missing_pad() : get_missing())
      : format(date.get_datenum());
  }

};


template<class TRAITS>
inline std::string
to_string(
  DateTemplate<TRAITS> date)
{
  return DateFormat::DEFAULT(date, false);
}


template<class TRAITS>
inline std::ostream&
operator<<(
  std::ostream& os,
  DateTemplate<TRAITS> date)
{
  os << to_string(date);
  return os;
}


}  // namespace date

//------------------------------------------------------------------------------

}  // namespace ora

//------------------------------------------------------------------------------
// Namespace imports
//------------------------------------------------------------------------------

namespace ora {

using date::DateFormat;
using daytime::DaytimeFormat;
using time::TimeFormat;

}  // namespace ora
