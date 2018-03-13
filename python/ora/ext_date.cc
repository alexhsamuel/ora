#include "py.hh"
#include "PyDate.hh"

//------------------------------------------------------------------------------

namespace ora {
namespace py {

void
set_up_dates(
  Module* const mod, 
  Module* const np_mod)
{
  PyDate<ora::date::Date>             ::add_to(*mod, "Date");
  PyDate<ora::date::Date16>           ::add_to(*mod, "Date16");
}


//------------------------------------------------------------------------------

}  // namespace ora
}  // namespace py

