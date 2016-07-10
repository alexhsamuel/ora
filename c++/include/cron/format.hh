#pragma once

#include <algorithm>
#include <iostream>
#include <sstream>
#include <stack>
#include <string>

#include "aslib/exc.hh"
#include "aslib/ptr.hh"
#include "aslib/string_builder.hh"
#include "cron/date.hh"
#include "cron/daytime_type.hh"
#include "cron/time_type.hh"
#include "cron/time_zone.hh"
#include "cron/types.hh"

namespace cron {

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

namespace _impl {

using namespace aslib;

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
  : pattern_(pattern)
  {
    static Parts parts{
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
    size_t const width = format(parts).length();
    invalid_ = std::string(width, ' ');
    invalid_.replace(0, 7, "INVALID");
    missing_ = std::string(width, ' ');
    missing_.replace(0, 7, "MISSING");
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

private:

  void format(StringBuilder&, Parts const&) const;

  std::string pattern_;
  std::string invalid_;
  std::string missing_;

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

  template<class TIME>
  std::string
  operator()(
    TIME const time, 
    TimeZone const& time_zone=UTC) 
    const 
  { 
    if (time.is_invalid())
      return get_invalid();
    else if (time.is_missing())
      return get_missing();
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
    std::string const& tz_name) 
    const 
  { 
    return operator()(time, get_time_zone(tz_name)); 
  }

  template<class TIME>
  std::string operator()(
    TIME const time, 
    _DisplayTimeZoneTag) 
    const
  {
    return operator()(time, *get_display_time_zone());
  }

};


template<class TRAITS>
inline std::string
to_string(
  TimeType<TRAITS> const time,
  TimeZone const& time_zone=UTC)
{
  return TimeFormat::DEFAULT(time, time_zone);
}


template<class TRAITS>
inline std::string
to_string(
  TimeType<TRAITS> const time,
  _DisplayTimeZoneTag)
{
  return to_string(time, *get_display_time_zone());
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
    DATE const date) 
    const 
  { 
    return
        date.is_invalid() ? get_invalid()
      : date.is_missing() ? get_missing()
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
  return DateFormat::DEFAULT(date);
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
    DaytimeTemplate<TRAITS> const daytime) 
    const 
  { 
    return
        daytime.is_invalid() ? get_invalid()
      : daytime.is_missing() ? get_missing()
      : operator()(get_hms(daytime));
  }

};


template<class TRAITS>
inline std::string
to_string(
  DaytimeTemplate<TRAITS> const daytime)
{
  return DaytimeFormat::DEFAULT(daytime);
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

}  // namespace cron

//------------------------------------------------------------------------------
// Namespace imports
//------------------------------------------------------------------------------

namespace cron {

using date::DateFormat;
using daytime::DaytimeFormat;
using time::TimeFormat;

}  // namespace cron


