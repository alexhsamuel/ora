// See: https://data.iana.org/time-zones/theory.html

#include <string>

#include "ora/posixtz.hh"
#include "ora/types.hh"

using ora::lib::FormatError;
using std::string;

//------------------------------------------------------------------------------

namespace {

string
parse_abbr(
  char const*& p)
{
  char const* const start = p;

  if (*p == '<') {
    // An angle-bracketed abbrev.
    while (*p != '>')
      if (*++p == 0)
        throw FormatError("unclosed <");
    // Skip the brackets.
    return string(start + 1, ++p - start - 2);
  }

  else {
    while (*p != 0 && *p != ',' && *p != '-' && !isdigit(*p))
      p++;
    if (p == start)
      throw FormatError("expected abbr");
    return string(start, p - start);
  }
}

int
parse_int(
  char const*& p)
{
  char const* const start = p;
  while (isdigit(*p))
    p++;
  if (p == start)
    throw FormatError("expected int");
  return std::atoi(string(start, p - start).c_str());
}

int
parse_sign(
  char const*& p)
{
  int sign = 1;
  if (*p == '+')
    ++p;
  else if (*p == '-') {
    sign = -1;
    ++p;
  }
  return sign;
}

int
parse_offset(
  char const*& p)
{
  int const sign = parse_sign(p);
  unsigned const hours = parse_int(p);
  unsigned mins = 0;
  unsigned secs = 0;
  if (*p == ':') {
    ++p;
    mins = parse_int(p);
    if (60 <= mins)
      throw FormatError("invalid mins");
    if (*p == ':') {
      ++p;
      secs = parse_int(p);
      if (60 <= secs)
        throw FormatError("invalid secs");
    }
  }
  return sign * (hours * 3600 + mins * 60 + secs);
}

ora::PosixTz::Transition
parse_transition(
  char const*& p)
{
  using Transition = ora::PosixTz::Transition;
  Transition trans;

  if (*p == 'J') {
    p++;
    trans.type = Transition::Type::JULIAN_WITHOUT_LEAP;
    auto& spec = trans.spec.julian;
    spec.ordinal = parse_int(p);
    if (spec.ordinal < 1 || 365 < spec.ordinal)
      throw FormatError("invalid Julian (without leap) ordinal");
  }

  else if (*p == 'M') {
    p++;
    trans.type = Transition::Type::GREGORIAN;
    auto& spec = trans.spec.gregorian;
    spec.month = parse_int(p);
    if (spec.month < 1 || 12 < spec.month)
      throw FormatError("invalid month");
    if (*p != '.')
      throw FormatError("expected . after month");
    ++p;
    spec.week = parse_int(p);
    if (spec.week < 1 || 5 < spec.week)
      throw FormatError("invalid week");
    if (*p != '.')
      throw FormatError("expected . after week");
    ++p;
    spec.weekday = parse_int(p);
    if (6 < spec.weekday)
      throw FormatError("invalid weekday");
  }

  else {
    trans.type = Transition::Type::JULIAN_WITH_LEAP;
    auto& spec = trans.spec.julian;
    spec.ordinal = parse_int(p);
    if (365 < spec.ordinal)
      throw FormatError("invalid Julian (with leap) ordinal");
  }

  // Optional offset.
  if (*p == '/') {
    p++;
    trans.ssm = parse_offset(p);
  }
  else
    // Default is 02:00:00.
    trans.ssm = 2 * 3600;

  return trans;
}

}  // anonymous namespace

//------------------------------------------------------------------------------

namespace ora {

PosixTz
parse_posix_time_zone(
  char const* const str)
{
  PosixTz tz;
  auto p = str;
  try {
    tz.std.abbreviation = parse_abbr(p);
    tz.std.offset = -parse_offset(p);
    if (*p != 0) {
      tz.dst.abbreviation = parse_abbr(p);
      tz.dst.offset =
        // No offset; assume one east of standard time.
        *p == ',' ? tz.std.offset + 3600
        // Explicit offset;
        : -parse_offset(p);
      if (*p != ',')
        throw FormatError("expected , before start");
      ++p;
      // DST start.
      tz.start = parse_transition(p);
      if (*p != ',')
        throw FormatError("expected , before end");
      ++p;
      // DST end.
      tz.end = parse_transition(p);
      if (*p != 0)
        throw FormatError("expected end of string");
    }
  }
  catch (FormatError& err) {
    throw FormatError(
      string(err.what()) + " at position " + std::to_string(p - str));
  }
  return tz;
}

//------------------------------------------------------------------------------

}  // namespace ora

