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
days_in_year =
"Returns the number of days in a year dates.\n"
"\n"
"  >>> days_in_year(2016)\n"
"  366\n"
"  >>> days_in_year(2017)\n"
"  365\n"
"\n"
"@signature\n"
"  days_in_year(year)\n"
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
"  >>> cron.from_local((date, daytime), 'UTC')\n"
"  Time(2016, 11, 11, 12, 0, 0.00000000, UTC)\n"
"\n"
"The same date and daytime in another time zone produces a different time:\n"
"\n"
"  >>> cron.from_local((date, daytime), 'US/Eastern')\n"
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
get_display_time_zone =
"Returns the display time zone.\n"
"\n"
"@signature\n"
"  get_display_time_zone()\n"
"\n"
;

doc_t
get_system_time_zone =
"Returns the system time zone.\n"
"\n"
"@signature\n"
"  get_system_time_zone()\n"
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

doc_t
set_display_time_zone =
"Sets the display time zone to `time_zone`.\n"
"\n"
"@signature\n"
"  set_display_time_zone(time_zone)\n"
"\n"
;

doc_t
now =
"Returns the current time.\n"
"\n"
"@signature\n"
"  now(Time=Time)\n"
"@param Time\n"
"  The time type to return.\n"
"\n"
;

doc_t
to_local =
"Converts a time to a local date and daytime representation.\n"
"\n"
"This is the primary function for converting actual times to local dates and\n"
"daytimes as humans usually represent them.  For example, UTC noon on 2016\n"
"Nov 11:\n"
"\n"
"  >>> time = Time(2016, 11, 11, 12, 0, 0, UTC)\n"
"  >>> cron.to_local(time, 'UTC')\n"
"  LocalTime(date=Date(2016, Oct, 10), daytime=Daytime(12, 0, 0.000000000000000))\n"
"\n"
"The same time converts to a different (date, daytime) pair in another time\n"
"zone:\n"
"\n"
"  >>> cron.to_local(time, 'US/Eastern')\n"
"  LocalTime(date=Date(2016, Nov, 11), daytime=Daytime(8, 0, 0.000000000000000))\n"
"\n"
"@signature\n"
"  to_local(time, time_zone, Date=Date, Daytime=Daytime)\n"
"@param Date\n"
"  The date type to produce.\n"
"@param Daytime\n"
"  The daytime type to produce.\n"
"@return\n"
"  A (date, daytime) tuple.  The fields may also be accessed as `date` and\n"
"  `daytime` attributes.\n"
"\n"
;

doc_t
to_local_datenum_daytick =
"Converts a time to a local representation of _datenum_ and _daytick_.\n"
"\n"
"  >>> time = Time(2016, 11, 11, 12, 0, 0, UTC)\n"
"  >>> datenum, daytick = cron.to_local_datenum_daytick(time, 'UTC')\n"
"\n"
"These could be converted into a date and daytime:\n"
"\n"
"  >>> Date.from_datenum(datenum)\n"
"  Date(2016, Oct, 10)\n"
"  >>> Daytime.from_daytick(daytick)\n"
"  Daytime(12, 0, 0.000000000000000)\n"
"\n"
"Use `to_local` to produce the date and daynum directly.\n"
"\n"
"@signature\n"
"  to_local_datenum_daytick(time, time_zone)\n"
"@return\n"
"  A (datenum, daytick) tuple.  The fields may also be accessed as `date` and\n"
"  `daytime` attributes.\n"
"\n"
;

doc_t
today =
"Returns the current date in a given time zone.\n"
"\n"
"The time zone is mandatory, as at any given time different parts of the\n"
"world are on two different dates.\n"
"\n"
"@signature\n"
"  today(time_zone, Date=Date)\n"
"@param Date\n"
"  The date type to return.\n"
"\n"
;

//------------------------------------------------------------------------------

}  // namespace docstring
}  // namesapce aslib

