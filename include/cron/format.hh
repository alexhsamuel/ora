#pragma once

#include <algorithm>
#include <iostream>
#include <sstream>
#include <stack>
#include <string>

#include "cron/date.hh"
#include "cron/daytime.hh"
#include "cron/time.hh"
#include "cron/time_zone.hh"
#include "cron/types.hh"
#include "exc.hh"
#include "ptr.hh"
#include "string_builder.hh"

namespace alxs {
namespace cron {

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
    // Find the width in characters by formatting a sample value.
    static DaytimeParts const daytime_parts{0, 0, 0};
    static TimeZoneParts const time_zone_parts{0, "", false};
    auto const width 
      = format(DATENUM_MIN, &daytime_parts, &time_zone_parts).length();

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
    Datenum                 datenum,
    DaytimeParts const*     daytime_parts,
    TimeZoneParts const*    time_zone_parts)
    const
  {
    StringBuilder sb;
    format(sb, datenum, daytime_parts, time_zone_parts);
    return sb.str();
  }

private:

  void format(StringBuilder&, Datenum, DaytimeParts const*, TimeZoneParts const*) const;

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
    return format(parts.datenum, &parts.daytime, &parts.time_zone); 
  }

  template<class TRAITS> 
  std::string
  operator()(
    TimeTemplate<TRAITS> time, 
    TimeZone const& tz) 
    const 
  { 
    return 
      time.is_valid() ? operator()(time.get_parts(tz))
      : time.is_missing() ? get_missing() 
      : get_invalid();
  }

  template<class TRAITS> 
  std::string
  operator()(
    TimeTemplate<TRAITS> time, 
    std::string const& tz_name) 
    const 
  { 
    return operator()(time, get_time_zone(tz_name)); 
  }

  template<class TRAITS> 
  std::string
  operator()(
    TimeTemplate<TRAITS> time) 
    const 
  { 
    return operator()(time, get_display_time_zone()); 
  }

};


template<class TRAITS>
inline std::string
to_string(
  TimeTemplate<TRAITS> time)
{
  return TimeFormat::get_default()(time);
}


template<class TRAITS>
inline std::ostream&
operator<<(
  std::ostream& os,
  TimeTemplate<TRAITS> time)
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
    static DateFormat const format("%Y-%m-%d", "INVALID   ", "MISSING   ");
    return format;
  }

  std::string
  operator()(
    datenum datenum) 
    const 
  { 
    return format(datenum, nullptr, nullptr); 
  }

  template<class TRAITS> 
  std::string
  operator()(
    DateTemplate<TRAITS> date) 
    const 
  { 
    return 
      date.is_valid() ? operator()(date.get_parts())
      : date.is_missing() ? get_missing()
      : get_invalid();
  }

};


template<class TRAITS>
inline std::string
to_string(
  DateTemplate<TRAITS> date)
{
  return DateFormat::get_default()(date);
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
    static DaytimeFormat const format("%H:%M:%S", "INVALID ", "MISSING ");
    return format;
  }

  std::string 
  operator()(
    DaytimeParts const& parts) 
    const 
  { 
    return format(DATENUM_INVALID, &parts, nullptr); 
  }

  template<class TRAITS> 
  std::string
  operator()(
    DaytimeTemplate<TRAITS> daytime) 
    const 
  { 
    return 
      daytime.is_valid() ? operator()(daytime.get_parts())
      : daytime.is_missing() ? get_missing()
      : get_invalid();
  }

};


template<class TRAITS>
inline std::string
to_string(
  DaytimeTemplate<TRAITS> daytime)
{
  return DaytimeFormat::get_default()(daytime);
}


template<class TRAITS>
inline std::ostream&
operator<<(
  std::ostream& os,
  DaytimeTemplate<TRAITS> daytime)
{
  os << to_string(daytime);
  return os;
}


//------------------------------------------------------------------------------

class TimeStrftimeFormat
{
public:

  struct Formatted
  {
    TimeStrftimeFormat const& format;
    TimeParts parts;

    operator std::string() { return alxs::to_string(*this); }  // FIXME
  };

  TimeStrftimeFormat(std::string const& pattern, bool extended=true) : pattern_(pattern), extended_(extended) {}
  TimeStrftimeFormat(char const* pattern, bool extended=true) : pattern_(pattern), extended_(extended) {}

  Formatted operator()(TimeParts const& parts) const { return {*this, parts}; }
  template<class TIME> Formatted operator()(TIME time) const { return {*this, time.get_parts(get_display_time_zone())}; }
  template<class TIME> Formatted operator()(TIME time, TimeZone const& tz) const { return {*this, time.get_parts(tz)}; }

  std::string get_pattern() const { return pattern_; }
  bool is_extended() const { return extended_; }

  void to_stream(std::ostream& os, TimeParts const& parts) const;

private:

  std::string pattern_;
  bool extended_;

};


inline std::ostream&
operator<<(
  std::ostream& os,
  TimeStrftimeFormat::Formatted const& formatted)
{
  formatted.format.to_stream(os, formatted.parts);
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
}  // namespace alxs

