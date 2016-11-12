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

//------------------------------------------------------------------------------

}  // namespace docstring
}  // namesapce aslib

