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

