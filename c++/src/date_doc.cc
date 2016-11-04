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


}  // namespace aslib

