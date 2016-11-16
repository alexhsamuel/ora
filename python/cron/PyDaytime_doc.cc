#include "PyDaytime_doc.hh"

namespace aslib {
namespace docstring {

//------------------------------------------------------------------------------

namespace pydaytime {

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

}  // namespace pydaytime

//------------------------------------------------------------------------------

}  // namespace docstring
}  // namespace aslib

