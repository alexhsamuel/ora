#pragma once

#include <algorithm>
#include <iostream>
#include <sstream>
#include <stack>
#include <string>

#include "ora/lib/exc.hh"
#include "ora/lib/ptr.hh"
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

extern std::string const& get_month_name(Month month);
extern Month parse_month_name(std::string const& str);
extern std::string const& get_month_abbr(Month month);
extern Month parse_month_abbr(std::string const& str);

/*
 * Returns the military / nautical time zone offset letter.
 *
 * Returns the one letter code of the [military time
 * zone](http://en.wikipedia.org/wiki/List_of_military_time_zones)
 * corresponding to an offset.  If there is no military time zone, returns an
 * unspecified nonalphabetic character.
 */
extern char get_time_zone_offset_letter(TimeZoneOffset);

extern std::string const& get_weekday_name(Weekday weekday);
extern Weekday parse_weekday_name(std::string const& str);
extern std::string const& get_weekday_abbr(Weekday weekday);
extern Weekday parse_weekday_abbr(std::string const& str);

//------------------------------------------------------------------------------

namespace {

inline void
format_second(
  StringBuilder& sb,
  Second const second,
  int const precision=-1,
  int const width=2,
  char const pad='0')
{
  // FIXME: Improve this logic.  See fixfmt.
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

  /*
   * Includes all the various parts that can be used for formatting.
   */
  struct Parts
  {
    FullDate      date            = {};
    bool          have_date       = false;
    HmsDaytime    daytime         = {};
    bool          have_daytime    = false;
    TimeZoneParts time_zone       = {};
    bool          have_time_zone  = false;
  };

  std::string 
  format(
    Parts const& parts)
    const
  {
    StringBuilder sb;
    format(sb, parts);
    return sb.str();
  }

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
      auto const parts = Parts{
        .date = FullDate{
          {YEAR_MIN, ORDINAL_MIN}, 
          {YEAR_MIN, MONTH_MIN, DAY_MIN}, 
          {YEAR_MIN, WEEK_MIN, WEEKDAY_MIN}}, 
        .have_date = true,
        .daytime = HmsDaytime{0, 0, 0}, 
        .have_daytime = true,
        .time_zone = TimeZoneParts{0, "", false},
        .have_time_zone = true,
      };
      width_ = (int) format(parts).length();

      // Truncate or pad the invalid and missing strings.
      invalid_pad_ = pad_trunc(invalid_, width_, ' ');
      missing_pad_ = pad_trunc(missing_, width_, ' ');
    }
  }

  void format(StringBuilder&, Parts const&) const;

  std::string const pattern_;
  std::string const invalid_;
  std::string const missing_;

  mutable int width_ = -1;
  mutable std::string invalid_pad_;
  mutable std::string missing_pad_;

};


}  // namespace _impl

//------------------------------------------------------------------------------

namespace time {

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
    else {
      // FIXME: Don't use daytick, which may lose precision.
      auto const ldd = to_local_datenum_daytick(time, time_zone);
      return format(Parts{
        .date           = datenum_to_full_date(ldd.datenum), 
        .have_date      = true,
        .daytime        = daytick_to_hms(ldd.daytick), 
        .have_daytime   = true,
        .time_zone      = ldd.time_zone, 
        .have_time_zone = true,
      });
    }
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
        tz = ora::get_system_time_zone();
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


extern inline void
format_iso_time(
  StringBuilder& sb,
  YmdDate const& date,
  HmsDaytime const& daytime,
  TimeZoneParts const& time_zone,
  int const precision,
  bool const compact=false,
  bool const capital=true,
  bool const military=false)
{
  sb.format(date.year, 4, '0');
  if (!compact)
    sb << '-';
  sb.format(date.month, 2, '0');
  if (!compact)
    sb << '-';
  sb.format(date.day, 2, '0');
  sb << (capital ? 'T' : 't');
  sb.format(daytime.hour, 2, '0');
  if (!compact)
    sb << ':';
  sb.format(daytime.minute, 2, '0');
  if (!compact)
    sb << ':';
  format_second(sb, daytime.second, precision);
  if (military)
    sb << get_time_zone_offset_letter(time_zone.offset);
  else
    format_iso_offset(sb, time_zone, !compact);
}


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
      : format(Parts{
          .date = datenum_to_full_date(date.get_datenum()),
          .have_date = true,
        });
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

namespace daytime {

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
    return format(Parts{
      .date = {}, .have_date = false,
      .daytime = hms, .have_daytime = true
    });
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
      : operator()(get_hms(daytime));
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

}  // namespace ora

//------------------------------------------------------------------------------
// Namespace imports
//------------------------------------------------------------------------------

namespace ora {

using date::DateFormat;
using daytime::DaytimeFormat;
using time::TimeFormat;

}  // namespace ora


