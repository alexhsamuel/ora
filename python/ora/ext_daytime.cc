#include "py.hh"
#include "PyDaytime.hh"

//------------------------------------------------------------------------------

namespace ora {
namespace py {

void
set_up_daytimes(
  Module* const mod, 
  Module* const np_mod)
{
  PyDaytime<ora::daytime::Daytime>    ::add_to(*mod, "Daytime");
  PyDaytime<ora::daytime::Daytime32>  ::add_to(*mod, "Daytime32");
  PyDaytime<ora::daytime::UsecDaytime>::add_to(*mod, "UsecDaytime");
}


//------------------------------------------------------------------------------

}  // namespace ora
}  // namespace py

