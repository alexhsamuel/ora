#include <Python.h>

#include "PyDate.hh"

using namespace alxs;
using namespace py;

//------------------------------------------------------------------------------

namespace {

Methods<Module> 
methods;

PyModuleDef
module_def{
  PyModuleDef_HEAD_INIT,
  "cron._ext",
  nullptr,
  -1,
  methods
};


}  // anonymous namespace

//------------------------------------------------------------------------------

PyMODINIT_FUNC
PyInit__ext(void)
{
  auto module = Module::Create(&module_def);

  try {
    alxs::PyDate<cron::DateTraits>::add_to(module, "Date");
    alxs::PyDate<cron::SmallDateTraits>::add_to(module, "SmallDate");
    return module.release();
  }
  catch (Exception) {
    return nullptr;
  }
}


