#pragma once

#include <string>

#include "ora/date_functions.hh"
#include "ora/date_type.hh"

namespace ora {

//------------------------------------------------------------------------------

namespace date {

extern bool parse_date_parts(
  char const*& pattern, char const*& string, FullDate& parts);

inline FullDate 
parse_date_parts(
  std::string const& pattern, 
  std::string const& string)
{
  FullDate parts;
  char const* p = pattern.c_str();
  char const* s = string.c_str();
  if (parse_date_parts(p, s, parts))
    return parts;
  else
    return {};
}


template<class DATE=Date>
inline DATE
parse(
  char const* const pattern,
  char const* const string)
{
  // FIXME: Accept ordinal date, week date instead.
  return ora::date::from_ymd<DATE>(parse_date_parts(pattern, string).ymd_date);
}


template<class DATE=Date>
inline DATE
parse(
  std::string const& pattern,
  std::string const& string)
{
  return parse(pattern.c_str(), string.c_str());
}


}  // namespace date

//------------------------------------------------------------------------------

namespace daytime {

extern bool parse_daytime_parts(
  char const*& pattern, char const*& string, HmsDaytime& hms);

}  // namespace daytime

//------------------------------------------------------------------------------

namespace time {

/*
 * If `letter_mode` is 1, parse a military time zone offset; if 0, parse
 * a full UTC offset.  If -1, accept either.
 */
extern bool parse_iso_time(
  char const*&, YmdDate&, HmsDaytime&, TimeZoneOffset&,
  int const letter_mode=-1, bool const compact=false);

// FIXME: Elsewhere.
struct TimeZoneInfo
{
  TimeZoneOffset offset = TIME_ZONE_OFFSET_INVALID;
  std::string name;
};

extern bool parse_time_parts(
  char const*& pattern, char const*& string, 
  FullDate& date, HmsDaytime& hms, TimeZoneInfo& tz);

}

//------------------------------------------------------------------------------

}  // namespace ora

