#include "PyDaytime_doc.hh"

namespace aslib {
namespace docstring {

//------------------------------------------------------------------------------

namespace pydaytime {

doc_t
type =
"A time of day.\n"
"\n"
"A daytime represents a time of day.  Daytime is a representation of a\n"
"specific time within a specific day in a specific location, as one might\n"
"read off an ordinary clock.  A daytime is an approximation to a specific\n"
"time of day, in the sense that a floating point number is an approximation\n"
"to a real number.\n"
"\n"
"An object of this daytime class can represent a time of day with\n"
"approximately %2$.1e second precision.\n"
"\n"
"# Constructor\n"
"\n"
"Construct a `%1$s` instance with any of these:\n"
"\n"
"- An instance of `%1$s`.\n"
"- An instance of another daytime type.\n"
"- The hour, minute, and second parts, as three arguments or a sequence.\n"
"- The hour and minute, as two arguments or a sequence.\n"
"- A double value, as _seconds since midnight_ (SSM).\n"
"- With no arguments, which constructs the value is `INVALID`.\n"
"\n"
;


doc_t
from_daytick =
"Constructs a daytime from _dayticks_.\n"
"\n"
"  >>> Daytime.from_daytick(6333186975989760000)\n"
"  Daytime(12, 30, 0.000000000000000)\n"
"\n"
"@signature\n"
"  from_daytick(daytick)\n"
"@see\n"
"  `cron.daytime`.\n"
"\n"
;

doc_t
from_hms =
"Constructs a daytime from hour, minute, and second.\n"
"\n"
"May be called with two or three arguments.\n"
"  >>> Daytime.from_hms(12, 30)\n"
"  Daytime(12, 30, 0.000000000000000)\n"
"  >>> Daytime.from_hms(12, 30, 10)\n"
"  Daytime(12, 30, 10.000000000000000)\n"
"  >>> Daytime.from_hms(12, 30, 45.6)\n"
"  Daytime(12, 30, 45.600000000000000)\n"
"\n"
"May also be called with a three-element sequence.\n"
"\n"
"  >>> Daytime.from_hms([12, 30, 45.6])\n"
"  Daytime(12, 30, 45.600000000000000)\n"
"\n"
"@signature\n"
"  from_hms(hour, minute, second=0)\n"
"\n"
;

doc_t
from_ssm =
"Constructs a daytime from _seconds since midnight_ (SSM).\n"
"\n"
"  >>> Daytime.from_ssm(12 * 60 * 60)\n"
"  Daytime(12, 0, 0.000000000000000)\n"
"  >>> Daytime.from_ssm(12345.678)\n"
"  Daytime(3, 25, 45.677999999999880)\n"
"\n"
"@signature\n"
"  from_ssm(ssm: float)\n"
"@see\n"
"  `cron.daytime`.\n"
"\n"
;

doc_t
daytick =
"The _daytick_, the number of 2**-47 seconds since midnight.\n"
"\n"
"  >>> Daytime(0, 0, 0).daytick\n"
"  0\n"
"  >>> Daytime(0, 0, 1).daytick\n"
"  140737488355328\n"
"\n"
;

doc_t
hour =
"The hour of the hour, minute, second representation.\n"
"\n"
"  >>> Daytime.MIDNIGHT.hour\n"
"  0\n"
"  >>> Daytime.MAX.hour\n"
"  23\n"
"\n"
;

}  // namespace pydaytime

//------------------------------------------------------------------------------

}  // namespace docstring
}  // namespace aslib

