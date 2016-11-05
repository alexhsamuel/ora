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
;


doc_t
datenum = 
"The \"datenum\" of this date.\n"
"\n"
"Cron performs date computations on \"datenums\", the number of days elapsed\n"
"since 0001 January 1.  (This is before the Gregorian calendar was adopted,\n"
"but we use the \"proleptic\" Gregorian calendar, which projects backward.)\n"
;


doc_t
day =
"The day of the month.\n"
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
"The date encoded as an 8-decimal digit \"YYYYMMDD\" integer.\n"
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

