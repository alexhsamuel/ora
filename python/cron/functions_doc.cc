#include "functions_doc.hh"

namespace aslib {
namespace docstring {

//------------------------------------------------------------------------------

doc_t
days_in_month =
"Returns the number of days in a month.\n"
"\n"
"The year must be provided to account for leap years.\n"
"\n"
"  >>> days_in_month(2000, Jan)\n"
"  31\n"
"  >>> days_in_month(2000, Feb)\n"
"  29\n"
"  >>> days_in_month(2001, Feb)\n"
"  28\n"
"\n"
"@signature\n"
"  days_in_month(year, month)\n"
"@rtype\n"
"  `int`\n"
"\n"
;

doc_t
from_local =
"Computes time from local time (date and daytime) and a time zone.\n"
"\n"
"This is the primary function from converting local times, as humans usually\n"
"represent them, to actual times.  For example, UTC noon on 2016 Nov 11:\n"
"\n"
"  >>> date = Date(2016, Nov, 11)\n"
"  >>> daytime = Daytime(12, 0, 0)\n"
"  >>> cron.from_local((date, daytime), \"UTC\")\n"
"  Time(2016, 11, 11, 12, 0, 0.00000000, UTC)\n"
"\n"
"The same date and daytime in another time zone produces a different time:\n"
"\n"
"  >>> cron.from_local((date, daytime), \"US/Eastern\")\n"
"  Time(2016, 11, 11, 17, 0, 0.00000000, UTC)\n"
"\n"
"@signature\n"
"  from_local(local_time, time_zone, first=True, Time=Time)\n"
"@param local_time\n"
"  A `(date, daytime)` pair containing the local time representation.\n"
"@param time_zone\n"
"  A time zone or time zone name.\n"
"@param first\n"
"  If the local time corresponds to two times, as can occur during a DST\n"
"  transition, specifies whether to use the first or second time.\n"
"@param Time\n"
"  The time type to return.\n"
"@raise RuntimeError\n"
"  The specified local time does not exist in the time zone.\n"
"\n"
;

doc_t
is_leap_year =
"Returns true if `year` is a leap year.\n"
"\n"
"@signature\n"
"  is_leap_year(year)\n"
"\n"
;

//------------------------------------------------------------------------------

}  // namespace docstring
}  // namesapce aslib

