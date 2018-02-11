#include <cstdint>

#include "ora/lib/math.hh"
#include "ora/parse.hh"

namespace ora {

using namespace ora::lib;
using std::string;

//------------------------------------------------------------------------------
// Helpers
//------------------------------------------------------------------------------

namespace {

#define TRY(expr) do { if (!(expr)) return false; } while (false)

template<class INT>
bool
parse_integer(
  char const*& p,
  INT& val)
{
  if (isdigit(*p)) {
    // First digit.
    val = *p++ - '0';
    // Remaining digits.
    while (true)
      if (isdigit(*p))
        if (   mul_overflow(val, 10, val)
            || add_overflow(val, *p - '0', val))
          // Overflowed.
          return false;
        else
          // Successfully processed the digit.
          ;
      else
        // Not a digit; we're done.
        return true;
  }
  else
    // No digits.
    return false;
}


template<>
bool
parse_integer(
  char const*& p,
  unsigned char& val)
{
  if (isdigit(*p)) 
    val = *p++ - '0';
  else
    return false;

  if (isdigit(*p))
    val = val * 10 + (*p++ - '0');
  else
    return true;

  if (isdigit(*p))
    val = val * 10 + (*p++ - '0');
  else
    return true;

  // OK if there are no more digits, and the value so far hasn't overflowed.
  return !isdigit(*p) && (*p & ~0xff) == 0;
}


template<>
bool
parse_integer(
  char const*& p,
  unsigned short& val)
{
  if (isdigit(*p)) 
    val = *p++ - '0';
  else
    return false;

  if (isdigit(*p))
    val = val * 10 + (*p++ - '0');
  else
    return true;

  if (isdigit(*p))
    val = val * 10 + (*p++ - '0');
  else
    return true;

  if (isdigit(*p))
    val = val * 10 + (*p++ - '0');
  else
    return true;

  if (isdigit(*p))
    val = val * 10 + (*p++ - '0');
  else
    return true;

  // OK if there are no more digits, and the value so far hasn't overflowed.
  return !isdigit(*p) && (*p & ~0xffff) == 0;
}


}  // anonymous namespace


//------------------------------------------------------------------------------
// Dates
//------------------------------------------------------------------------------

namespace date {

bool
parse_date_parts(
  char const*& p,
  char const*& s,
  FullDate& parts)
{
  while (true)
    if (*p == 0 && *s == 0)
      // Completed successfully.
      return true;
    else if (*p == '%') {
      ++p;
      if (*p == '%')
        // Literal '%'.
        if (*s == '%') {
          ++p;
          ++s;
          continue;
        }
        else
          // Didn't match %.
          return false;

      // Modifiers mods;
      // parse_modifiers(p, mods);

      switch (*p) {
      case 'd':
        TRY(parse_integer(s, parts.ymd_date.day));
        break;
      case 'm':
        TRY(parse_integer(s, parts.ymd_date.month));
        break;
      case 'Y':
        TRY(parse_integer(s, *(unsigned short*) &parts.ymd_date.year));
        break;
      }
      ++p;
    }
    else if (*p == *s) {
      ++p;
      ++s;
    }
    else
      return false;
}


}  // namespace date


//------------------------------------------------------------------------------

}  // namespace ora

