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
#include "cron/daytime.hh"
#include "cron/time.hh"
#include "cron/time_zone.hh"
#include "cron/types.hh"

namespace cron {

using namespace aslib;

//------------------------------------------------------------------------------

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
    static DateParts const date_parts{0, 0, 0, 0, 0, 0, 0};
    static HmsDaytime const daytime_parts{0, 0, 0};
    static TimeZoneParts const time_zone_parts{0, "", false};
    size_t const width = format(&date_parts, &daytime_parts, &time_zone_parts).length();
    invalid_ = std::string(width, ' ');
    invalid_.replace(0, 7, "INVALID");
    missing_ = std::string(width, ' ');
    missing_.replace(0, 7, "MISSING");
  }

  std::string const& get_pattern() const { return pattern_; }
  std::string const& get_invalid() const { return invalid_; }
  std::string const& get_missing() const { return missing_; }

protected:

  std::string 
  format(
    DateParts const*     date_parts, 
    HmsDaytime const*    daytime_parts, 
    TimeZoneParts const* time_zone_parts)
    const
  {
    StringBuilder sb;
    format(sb, date_parts, daytime_parts, time_zone_parts);
    return sb.str();
  }

private:

  void format(StringBuilder&, DateParts const*, HmsDaytime const*, TimeZoneParts const*) const;

  std::string pattern_;
  std::string invalid_;
  std::string missing_;

};


//------------------------------------------------------------------------------

class TimeFormat
  : public Format
{
public:

  static TimeFormat const ISO_LOCAL_BASIC;
  static TimeFormat const ISO_LOCAL_EXTENDED;
  static TimeFormat const ISO_UTC_BASIC;
  static TimeFormat const ISO_UTC_EXTENDED;
  static TimeFormat const ISO_ZONE_BASIC;
  static TimeFormat const ISO_ZONE_EXTENDED;

  using Format::Format;

  static TimeFormat const& 
  get_default() 
  { 
    static TimeFormat const format(
      "%Y-%m-%d %H:%M:%S %~Z",
      "INVALID                ",
      "MISSING                ");
    return format;
  }

  std::string
  operator()(
    TimeParts const& parts) 
    const 
  { 
    return format(&parts.date, &parts.daytime, &parts.time_zone); 
  }

  template<class TRAITS> 
  std::string
  operator()(
    time::TimeTemplate<TRAITS> time, 
    TimeZone const& tz) 
    const 
  { 
    return 
      time.is_valid() ? operator()(get_parts(time, tz))
      : time.is_missing() ? get_missing() 
      : get_invalid();
  }

  template<class TRAITS> 
  std::string
  operator()(
    time::TimeTemplate<TRAITS> time, 
    std::string const& tz_name) 
    const 
  { 
    return operator()(time, get_time_zone(tz_name)); 
  }

  template<class TRAITS> 
  std::string
  operator()(
    time::TimeTemplate<TRAITS> time) 
    const 
  { 
    return operator()(time, *get_display_time_zone()); 
  }

};


template<class TRAITS>
inline std::string
to_string(
  time::TimeTemplate<TRAITS> time)
{
  return TimeFormat::get_default()(time);
}


template<class TRAITS>
inline std::ostream&
operator<<(
  std::ostream& os,
  time::TimeTemplate<TRAITS> time)
{
  os << to_string(time);
  return os;
}


//------------------------------------------------------------------------------

class DateFormat
  : public Format
{
public:

  static DateFormat const ISO_CALENDAR_BASIC;
  static DateFormat const ISO_CALENDAR_EXTENDED;
  static DateFormat const ISO_ORDINAL_BASIC;
  static DateFormat const ISO_ORDINAL_EXTENDED;
  static DateFormat const ISO_WEEK_BASIC;
  static DateFormat const ISO_WEEK_EXTENDED;

  using Format::Format;

  static DateFormat const& 
  get_default() 
  { 
    // Use representations for invalid and missing that are the same length.
    static DateFormat const format(
      "%Y-%m-%d", 
      "INVALID   ", 
      "MISSING   ");
    return format;
  }

  std::string
  operator()(
    DateParts const& parts) 
    const 
  { 
    return format(&parts, nullptr, nullptr); 
  }

  template<class DATE> 
  std::string
  operator()(
    DATE const date) 
    const 
  { 
    return 
      date.is_valid() ? operator()(datenum_to_parts(date.get_datenum()))
      : date.is_missing() ? get_missing()
      : get_invalid();
  }

};


template<class TRAITS>
inline std::string
to_string(
  date::DateTemplate<TRAITS> date)
{
  return DateFormat::get_default()(date);
}


template<class TRAITS>
inline std::ostream&
operator<<(
  std::ostream& os,
  date::DateTemplate<TRAITS> date)
{
  os << to_string(date);
  return os;
}


//------------------------------------------------------------------------------

class DaytimeFormat
  : public Format
{
public:
  
  static DaytimeFormat const ISO_BASIC;
  static DaytimeFormat const ISO_EXTENDED;
  static DaytimeFormat const ISO_BASIC_MSEC;
  static DaytimeFormat const ISO_EXTENDED_MSEC;
  static DaytimeFormat const ISO_BASIC_USEC;
  static DaytimeFormat const ISO_EXTENDED_USEC;
  static DaytimeFormat const ISO_BASIC_NSEC;
  static DaytimeFormat const ISO_EXTENDED_NSEC;

  using Format::Format;

  static DaytimeFormat const& 
  get_default() 
  { 
    // Use representations for invalid and missing that are the same length.
    static DaytimeFormat const format(
      "%H:%M:%S", 
      "INVALID ", 
      "MISSING ");
    return format;
  }

  std::string 
  operator()(
    HmsDaytime const& parts) 
    const 
  { 
    return format(nullptr, &parts, nullptr); 
  }

  template<class TRAITS> 
  std::string
  operator()(
    daytime::DaytimeTemplate<TRAITS> const daytime) 
    const 
  { 
    return 
        daytime.is_valid() ? operator()(get_hms(daytime))
      : daytime.is_missing() ? get_missing()
      : get_invalid();
  }

};


template<class TRAITS>
inline std::string
to_string(
  daytime::DaytimeTemplate<TRAITS> const daytime)
{
  return DaytimeFormat::get_default()(daytime);
}


template<class TRAITS>
inline std::ostream&
operator<<(
  std::ostream& os,
  daytime::DaytimeTemplate<TRAITS> const daytime)
{
  os << to_string(daytime);
  return os;
}


//------------------------------------------------------------------------------
// Functions
//------------------------------------------------------------------------------

extern std::string const& get_month_name(Month month);
extern Month parse_month_name(std::string const& str);
extern std::string const& get_month_abbr(Month month);
extern Month parse_month_abbr(std::string const& str);

extern std::string const& get_weekday_name(Weekday weekday);
extern Weekday parse_weekday_name(std::string const& str);
extern std::string const& get_weekday_abbr(Weekday weekday);
extern Weekday parse_weekday_abbr(std::string const& str);

//------------------------------------------------------------------------------

}  // namespace cron


