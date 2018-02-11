#include <cstdint>

#include "ora/format.hh"
#include "ora/lib/math.hh"
#include "ora/parse.hh"

namespace ora {

using namespace ora::lib;
using std::string;

//------------------------------------------------------------------------------
// Helpers
//------------------------------------------------------------------------------

namespace {

#define TRY(stmt) do { if (!(stmt)) return false; } while(false)

/*
 * Skips over formatting modifiers; see `parse_modifiers`.
 */
inline bool
skip_modifiers(
  char const*& p)
{
  bool decimal = false;

  for (; *p != 0; ++p)
    switch (*p) {
    case '.':
      if (decimal)
        // Two decimal points.
        return false;
      else
        decimal = true;
      break;

    case '0': case '1': case '2': case '3': case '4':
    case '5': case '6': case '7': case '8': case '9':
    case '^': case '_': case '~':
      break;

    case '#':
      if (*++p == 0)
        // Must be followed by another character.
        return false;
      break;

    default:
      return true;
    }

  // Unrechable.
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

  for (size_t i = 0; i < MAX_DIGITS - 1; ++i)
    if (isdigit(*p))
      val = val * 10 + (*p++ - '0');
    else
      return val;

  return val;
}


inline bool
parse_d(
  char const*& s,
  Day& day)
{
  auto const i = parse_unsigned<2>(s);
  if (day_is_valid(i)) {
    day = i;
    return true;
  }
  else
    return false;
}


inline bool
parse_m(
  char const*& s,
  Month& month)
{
  auto const i = parse_unsigned<2>(s);
  if (month_is_valid(i)) {
    month = i;
    return true;
  }
  else
    return false;
}


inline bool
parse_Y(
  char const*& s,
  Year& year)
{
  auto const i = parse_unsigned<4>(s);
  if (year_is_valid(i)) {
    year = i;
    return true;
  }
  else
    return false;
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

      skip_modifiers(p);

      switch (*p) {
      case 'B': TRY(parse_month_name(s, parts.ymd_date.month)); break;
      case 'b': TRY(parse_month_abbr(s, parts.ymd_date.month)); break;
      case 'd': TRY(parse_d(s, parts.ymd_date.day)); break;
      case 'm': TRY(parse_m(s, parts.ymd_date.month)); break;
      case 'Y': TRY(parse_Y(s, parts.ymd_date.year)); break;

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

