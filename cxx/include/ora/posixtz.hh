#pragma once

#include <ostream>
#include <string>

namespace ora {

//------------------------------------------------------------------------------

struct PosixTz
{
  struct Transition
  {
    enum Type {
      JULIAN_WITHOUT_LEAP,
      JULIAN_WITH_LEAP,
      GREGORIAN,
    };

    struct Julian
    {
      unsigned short ordinal;
    };

    struct Gregorian
    {
      unsigned char month;
      unsigned char week;
      unsigned char weekday;
    };

    Type type;
    union {
      Julian julian;
      Gregorian gregorian;
    } spec;
    unsigned int ssm;
  };

  struct Type
  {
    std::string abbreviation;
    int offset;
  };

  // Standard time.
  struct Type std;

  // Daylight Saving Time.  `dst.abbreviation == ""` if none.
  struct Type dst;

  // DST start and end; unused if no DST.
  Transition start;
  Transition end;
};


PosixTz parse_posix_time_zone(char const* str);

inline std::ostream&
operator<<(
  std::ostream& os,
  PosixTz::Transition const& trans)
{
  switch (trans.type) {
  case PosixTz::Transition::Type::JULIAN_WITHOUT_LEAP:
    os << "Julian (no Feb 29) " << trans.spec.julian.ordinal;
    break;
  case PosixTz::Transition::Type::JULIAN_WITH_LEAP:
    os << "Julian (with Feb 29) " << trans.spec.julian.ordinal;
    break;
  case PosixTz::Transition::Type::GREGORIAN:
    os << "Gregorian month=" << (unsigned) trans.spec.gregorian.month
       << " week=" << (unsigned) trans.spec.gregorian.week
       << " weekday=" << (unsigned) trans.spec.gregorian.weekday;
    break;
  }
  os << " SSM=" << trans.ssm;
  return os;
}

inline std::ostream&
operator<<(
  std::ostream& os,
  PosixTz const& tz)
{
  os << "Standard time: " << tz.std.abbreviation
     << " offset=" << tz.std.offset << "\nDaylight Saving Time: ";
  if (tz.dst.abbreviation == "")
    os << "none\n";
  else
    os << tz.dst.abbreviation << " offset=" << tz.dst.offset << "\n"
       << "  start: " << tz.start << "\n"
       << "  end:   " << tz.end << "\n";
  return os;
}


//------------------------------------------------------------------------------

}  // namespace ora

