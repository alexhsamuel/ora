#pragma once

#include <cstring>
#include <string>

namespace alxs {

//------------------------------------------------------------------------------
// Types
//------------------------------------------------------------------------------

typedef std::pair<std::string, std::string> 
StringPair;


//------------------------------------------------------------------------------
// Constants
//------------------------------------------------------------------------------

static char const* const 
WHITESPACE 
  = " \t\n\r";


//------------------------------------------------------------------------------
// Functions
//------------------------------------------------------------------------------

template<typename T> 
inline std::string
to_string(
  T val)
{
  std::stringstream os;
  os << val;
  return os.str();
}


template<> 
inline std::string
to_string(
  char const* str)
{
  return std::string(str);
}


template<> 
inline std::string
to_string(
  struct timeval val)
{
  return to_string(val.tv_sec + 1e-6 * val.tv_usec);
}


//------------------------------------------------------------------------------

inline std::string
strip(
  std::string const& text,
  char const* space=WHITESPACE)
{
  std::string::size_type const start = text.find_first_not_of(space);
  std::string::size_type const end   = text.find_last_not_of(space);
  return start == std::string::npos ? std::string() : text.substr(start, end - start + 1);
}


inline StringPair
split1(
  std::string const& text,
  char const* delim=WHITESPACE)
{
  std::string::size_type const pos0 = text.find_first_of(delim);
  std::string::size_type const pos1 = text.find_first_not_of(delim, pos0);
  if (pos0 == std::string::npos || pos1 == std::string::npos)
    return {text, ""};
  else
    return {text.substr(0, pos0), text.substr(pos1)};
}


//------------------------------------------------------------------------------

}  // namespace alxs

