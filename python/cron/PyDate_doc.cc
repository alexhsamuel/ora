namespace aslib {

extern char const* const
PyDate_doc = 
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


extern auto const
PyDate_datenum_doc = 
"The \"datenum\" of this date.\n"
"\n"
"Cron performs date computations on \"datenums\", the number of days elapsed\n"
"since 0001 January 1.  (This is before the Gregorian calendar was adopted,\n"
"but we use the \"proleptic\" Gregorian calendar, which projects backward.)\n"
;


extern auto const
PyDate_day_doc =
"The day of the month.\n"
;


extern auto const
PyDate_invalid_doc =
"True if this is `INVALID`.\n"
;


extern auto const
PyDate_missing_doc = 
"True if this is `MISSING`.\n"
;


extern auto const
PyDate_month_doc =
"The calendar month of which this date is part.\n"
;


extern auto const
PyDate_offset_doc =
"The type-specific offset used as the internal representation of this date.\n"
;


extern auto const
PyDate_ordinal_doc =
"The ordinal date: the 1-indexed day of the year.\n"
;


extern auto const
PyDate_ordinal_date_doc =
"A (year, ordinal) object representing the ordinal date.\n"
;


extern auto const
PyDate_valid_doc =
"True if this date is not `MISSING` or `INVALID`.\n"
;


extern auto const
PyDate_week_doc =
"The week number of the ISO-8601 week date.\n"
;


extern auto const
PyDate_week_date_doc =
"A (week_year, week, weekday) object containing the ISO-8601 week date.\n"
;


extern auto const
PyDate_week_year_doc =
"The year of the ISO-8601 week date.\n"
"\n"
"Note that this is not necessarily the same as the ordinary `year`.\n"
;


extern auto const
PyDate_weekday_doc =
"The day of the week.\n"
;


extern auto const
PyDate_year_doc =
"The year.\n"
"\n"
"This is the year of the conventional (year, month, day) representation,\n"
"not of the ISO-8601 week date representation.\n";
;


extern auto const
PyDate_ymdi_doc =
"The date encoded as an 8-decimal digit \"YYYYMMDD\" integer.\n"
;


extern auto const
PyDate_ymd_doc =
"An object containing the (year, month, day) date components.\n"
;


}  // namespace aslib

