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


template<size_t MAX_DIGITS>
inline int
parse_unsigned(
  char const*& p)
{
  int val;

  if (isdigit(*p)) 
    val = *p++ - '0';
  else
    return -1;

#pragma unroll 9
  for (size_t i = 0; i < MAX_DIGITS - 1; ++i)
    if (isdigit(*p))
      val = val * 10 + (*p++ - '0');
    else
      return val;

  return val;
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
      case 'd': {
        auto const i = parse_unsigned<2>(s);
        if (day_is_valid(i))
          parts.ymd_date.day = i;
        else
          return false;
      } break;

      case 'm': {
        auto const i = parse_unsigned<2>(s);
        if (month_is_valid(i))
          parts.ymd_date.month = i;
        else
          return false;
      } break;

      case 'Y': {
        auto const i = parse_unsigned<4>(s);
        if (year_is_valid(i))
          parts.ymd_date.year = i;
        else
          return false;
      } break;

      default:
        return false;
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

