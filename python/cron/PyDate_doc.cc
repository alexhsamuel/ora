#include "PyDate_doc.hh"

namespace aslib {
namespace docstring {

//------------------------------------------------------------------------------

namespace pydate {

doc_t
type = 
"A calendar date.\n"
"\n"
"In a specific location, a calendar date corresponds to a period usually, but\n"
"not always, 24 hours long.  A calendar date by itself does not represent any\n"
"specific time or interval of time.\n"
"\n"
"An object of this date class can represent any date between %2$s and %3$s,\n"
"inclusive.\n"
"\n"
"# Constructor\n"
"\n"
"Construct a `%1$s` instance with any of these:\n"
"\n"
"- An instance of `%1$s`.\n"
"- An instance of another date type.\n"
"- The strings `\"MIN\"` or `\"MAX\"`.\n"
"- An ISO-8859-formatted date string, e.g. `\"1989-12-31\"`.\n"
"- Two arguments or a two-element sequence (year, ordinal day).\n"
"- Two arguments or a three-element sequence (year, month, day).\n"
"- A YYYYMMDD-encoded integer.\n"
"\n"
"The following all construct the same date:\n"
"\n"
"  >>> %1$s(2004, 11, 2)\n"
"  %1$s(2004, 11, 2)\n"
"  >>> %1$s([2004, 11, 2])\n"
"  %1$s(2004, 11, 2)\n"
"  >>> %1$s(%1$s(2004, 11, 2))\n"
"  %1$s(2004, 11, 2)\n"
"  >>> %1$s(2004, 307)\n"
"  %1$s(2004, 11, 2)\n"
"  >>> %1$s([2004, 307])\n"
"  %1$s(2004, 11, 2)\n"
"  >>> %1$s(20041102)\n"
"  %1$s(2004, 11, 2)\n"
"  >>> %1$s(\"2004-11-02\")\n"
"  %1$s(2004, 11, 2)\n"
"\n"
"# Exceptions\n"
"\n"
"Methods may raise these exceptions:\n"
"\n"
"- `TypeError`: The number of arguments is wrong, or an arguments could not\n"
"  be converted to the right type.\n"
"- `ValueError`: An argument's value is invalid, e.g. month number 15.\n"
"- `OverflowError`: The method produced a date that it out of range for\n"
"  `%1$s`.\n"
"\n"
;


doc_t
datenum = 
"The _datenum_ of this date.\n"
"\n"
"  >>> Date(2004, 11, 2).datenum\n"
"  731886\n"
"\n"
;


doc_t
day =
"The day of the month.\n"
;


doc_t
from_datenum =
"Constructs a date from a _datenum_.\n"
"\n"
"  >>> Date.from_datenum(731886)\n"
"  Date(2004, Nov, 2)\n"
"\n"
"@signature\n"
"  from_datenum(datenum)\n"
"\n"
;


doc_t
from_iso_date =
"Constructs a date from an ISO-formatted date string.\n"
"\n"
"  >>> Date.from_iso_date(\"2014-11-02\")\n"
"  Date(2014, Nov, 2)\n"
"\n"
"@signature\n"
"  from_iso_date(iso_date)\n"
"\n"
;


doc_t
from_offset =
"Constructs a date from an offset.\n"
"\n"
"The offset is an implementation detail of the type.\n"
"\n"
"@signature\n"
"  from_offset(offset)\n"
"@raise ValueError\n"
"  `offset` is not a valid offset.\n"
"\n"
;


doc_t 
from_ordinal_date =
"Constructs a date from an ordinal date.\n"
"\n"
"  >>> Date.from_ordinal_date(2000, 1)\n"
"  Date(2000, Jan, 1)\n"
"  >>> Date.from_ordinal_date(2004, 307)\n"
"  Date(2004, Nov, 2)\n"
"\n"
"The year and ordinal may also be given as a two-element sequence.\n"
"\n"
"  >>> Date.from_ordinal_date([2004, 307])\n"
"  Date(2004, Nov, 2)\n"
"\n"
"@signature\n"
"  from_ordinal_date(year, ordinal)\n"
"@param ordinal\n"
"  The one-indexed day ordinal within the year.\n"
"\n"
;


doc_t
from_week_date =
"Constructs a date from an ISO-8601 week date.\n"
"\n"
"  >>> Date.from_week_date(2004, 45, Tue)\n"
"  Date(2004, Nov, 2)\n"
"\n"
"The components may also be given as a three-element sequence.\n"
"\n"
"  >>> Date.from_week_date([2004, 45, Tue])\n"
"  Date(2004, Nov, 2)\n"
"\n"
":warning: The week year is not the same as the ordinary year; see \n"
"`cron.date`.\n"
"\n"
"@signature\n"
"  from_week_date(week_year, week, weekday)\n"
"\n"
;


doc_t
from_ymd =
"Constructs a date from a year, month, and day.\n"
"\n"
"  >>> Date.from_ymd(2004, Nov, 2)\n"
"  Date(2004, Nov, 2)\n"
"\n"
"The components may also be given as a three-element sequence.\n"
"\n"
"  >>> Date.from_ymd([2004, Nov, 2])\n"
"  Date(2004, Nov, 2)\n"
"\n"
"@signature\n"
"  from_ymd(year, month, day)\n"
"\n"
;


doc_t
from_ymdi =
"Constructs a date from a _YMDI_ (YYYYMMDD-encoded integer).\n"
"\n"
"  >>> Date.from_ymdi(20041102)\n"
"  Date(2004, Nov, 2)\n"
"\n"
"@signature\n"
"  from_ymdi(ymdi)\n"
"\n"
;


doc_t
invalid =
"True if this is `INVALID`.\n"
;


doc_t
missing = 
"True if this is `MISSING`.\n"
;


doc_t
month =
"The calendar month of which this date is part.\n"
;


doc_t
offset =
"The type-specific offset used as the internal representation of this date.\n"
;


doc_t
ordinal =
"The ordinal date: the 1-indexed day of the year.\n"
;


doc_t
ordinal_date =
"A (year, ordinal) object representing the ordinal date.\n"
;


doc_t
valid =
"True if this date is not `MISSING` or `INVALID`.\n"
;


doc_t
week =
"The week number of the ISO-8601 week date.\n"
;


doc_t
week_date =
"A (week_year, week, weekday) object containing the ISO-8601 week date.\n"
;


doc_t
week_year =
"The year of the ISO-8601 week date.\n"
"\n"
"Note that this is not necessarily the same as the ordinary `year`.\n"
;


doc_t
weekday =
"The day of the week.\n"
;


doc_t
year =
"The year.\n"
"\n"
"This is the year of the conventional (year, month, day) representation,\n"
"not of the ISO-8601 week date representation.\n";
;


doc_t
ymdi =
"The date as a _YMDI_ (YYYYMMDD-encoded integer).\n"
;


doc_t
ymd =
"An object containing the (year, month, day) date components.\n"
;


}  // namespace pydate


//------------------------------------------------------------------------------

namespace ymddate {

doc_t
type =
"A (year, month, date) tuple containing components of the Gregorian date.\n"
"\n"
"  >>> ymd = YmdDate((1973, Dec, 3))\n"
"  >>> y, m, d = ymd\n"
"  >>> list(ymd)\n"
"  [1973, Month.Dec, 3]\n"
"\n"
"The components are also accessible as attributes.\n"
"\n"
"  >>> ymd.year\n"
"  1973\n"
"\n"
;

}  // namespace ymddate

//------------------------------------------------------------------------------

}  // namespace docstring
}  // namespace aslib

