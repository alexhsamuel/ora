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
  while (*p != 0 && *p != ',' && *p != '-' && !isdigit(*p))
    p++;
  if (p == start)
    throw FormatError("expected abbr");
  return string(start, p - start);
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

int parse_signed_int(
  char const*& p)
{
  bool neg = false;
  if (*p == '+')
    ++p;
  else if (*p == '-') {
    neg = true;
    ++p;
  }
  int val = parse_int(p);
  return neg ? -val : val;
}

ora::HmsDaytime
parse_daytime(
  char const*& p)
{
  ora::HmsDaytime daytime{(ora::Hour) parse_signed_int(p), 0, 0};
  if (daytime.hour < -24 || 24 < daytime.hour)
    throw FormatError("invalid hour");
  if (*p == ':') {
    daytime.minute = parse_int(p);
    if (daytime.minute > 59)
      throw FormatError("invalid minute");
    if (*p == ':') {
      daytime.second = parse_int(p);
      if (daytime.second > 59)
        throw FormatError("invalid second");
    }
  }
  return daytime;
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
    ++p;
    auto const hms = parse_daytime(p);
    trans.ssm = hms.hour * 3600 + hms.minute * 60 + hms.second;
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
    tz.std.offset = -3600 * parse_signed_int(p);
    if (*p != 0) {
      tz.dst.abbreviation = parse_abbr(p);
      tz.dst.offset =
        // No offset; assume one east of standard time.
        *p == ',' ? tz.std.offset + 3600
        // Explicit offset;
        : -3600 * parse_int(p);
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

