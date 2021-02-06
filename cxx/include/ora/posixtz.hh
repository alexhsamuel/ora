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
    string abbreviation;
    int offset;
    Transition transition;
  };

};


PosixTz parse(std::string str);

inline std::ostream
operator<<(
  std::ostream& os,
  PosixTz const& tz)
{
  
  return os;
}


//------------------------------------------------------------------------------

}  // namespace ora

