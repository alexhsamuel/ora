#include <Python.h>

#include "PyDate.hh"
#include "PyTime.hh"

using namespace alxs;
using namespace py;

//------------------------------------------------------------------------------

/** Adds functions from functions.cc.  */
extern Methods<Module>& add_functions(Methods<Module>&);

namespace {

Methods<Module> 
methods;

PyModuleDef
module_def{
  PyModuleDef_HEAD_INIT,
  "cron._ext",
  nullptr,
  -1,
  add_functions(methods)
};


}  // anonymous namespace

//------------------------------------------------------------------------------

PyMODINIT_FUNC
PyInit__ext(void)
{
  auto module = Module::Create(&module_def);

  try {
    alxs::PyDate<cron::Date>        ::add_to(module, "Date");
    alxs::PyDate<cron::SmallDate>   ::add_to(module, "SmallDate");

    alxs::PyTime<cron::Time>        ::add_to(module, "Time");
    alxs::PyTime<cron::SmallTime>   ::add_to(module, "SmallTime");
    alxs::PyTime<cron::NsecTime>    ::add_to(module, "NsecTime");
    alxs::PyTime<cron::Unix32Time>  ::add_to(module, "Unix32Time");
    alxs::PyTime<cron::Unix64Time>  ::add_to(module, "Unix64Time");

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


