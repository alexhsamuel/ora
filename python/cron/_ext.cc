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
    alxs::PyDate<cron::Date>::add_to(module, "Date");
    alxs::PyDate<cron::SmallDate>::add_to(module, "SmallDate");

    StructSequenceType* const parts_type = get_date_parts_type();
    module->AddObject(parts_type->tp_name, (PyObject*) parts_type);

    module->AddObject("DATENUM_MIN" , Long::FromLong(cron::DATENUM_MIN));
    module->AddObject("DATENUM_LAST", Long::FromLong(cron::DATENUM_LAST));
    module->AddObject("DATENUM_MAX" , Long::FromLong(cron::DATENUM_MAX));

    return module.release();
  }
  catch (Exception) {
    return nullptr;
  }
}


