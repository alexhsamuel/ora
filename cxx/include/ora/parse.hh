#pragma once

#include <string>

#include "ora/date_functions.hh"
#include "ora/date_type.hh"

namespace ora {

//------------------------------------------------------------------------------

namespace date {

extern bool parse_date_parts(char const*& pattern, char const*& string, FullDate& parts);

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

}  // namespace ora

