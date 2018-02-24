#pragma once

#include <cstring>
#include <sstream>
#include <string>

namespace ora {
namespace lib {

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

template<class T> 
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


/*
 * Right-pads or truncates `str` to `width` characters.
 */
inline std::string
pad_trunc(
  std::string const& str,
  size_t const width,
  char const pad)
{
  auto const len = str.length();
  if (len == width)
    return str;
  else if (len < width) {
    auto padded = str;
    padded.append(width - len, pad);
    return padded;
  }
  else
    return str.substr(0, width);
}


//------------------------------------------------------------------------------

}  // namespace lib
}  // namespace ora

