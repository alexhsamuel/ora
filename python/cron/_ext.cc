#include <Python.h>
#include <datetime.h>

#include "PyDate.hh"
#include "PyDaytime.hh"
#include "PyTime.hh"
#include "PyTimeZone.hh"

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

// FIXME
extern void init_date_dtype(Module*);

PyMODINIT_FUNC
PyInit__ext(void)
{
  // Set up the C API to the standard library datetime module.
  if (PyDateTimeAPI == nullptr) {
    PyDateTime_IMPORT;
    assert(PyDateTimeAPI != nullptr);
  }

  auto module = Module::Create(&module_def);

  try {
    alxs::PyDate<cron::Date>            ::add_to(module, "Date");
    alxs::PyDate<cron::SmallDate>       ::add_to(module, "SmallDate");

    alxs::PyDaytime<cron::Daytime>      ::add_to(module, "Daytime");
    alxs::PyDaytime<cron::SmallDaytime> ::add_to(module, "SmallDaytime");

    alxs::PyTime<cron::Time>            ::add_to(module, "Time");
    alxs::PyTime<cron::SmallTime>       ::add_to(module, "SmallTime");
    alxs::PyTime<cron::NsecTime>        ::add_to(module, "NsecTime");
    alxs::PyTime<cron::Unix32Time>      ::add_to(module, "Unix32Time");
    alxs::PyTime<cron::Unix64Time>      ::add_to(module, "Unix64Time");

    alxs::PyTimeZone                    ::add_to(module, "TimeZone");

    StructSequenceType* const parts_type = get_date_parts_type();
    module->AddObject(parts_type->tp_name, (PyObject*) parts_type);

    module->AddObject("DATENUM_MIN" , Long::FromLong(cron::DATENUM_MIN));
    module->AddObject("DATENUM_MAX" , Long::FromLong(cron::DATENUM_MAX));
    module->AddObject("MIDNIGHT"    , PyDaytimeDefault::create(PyDaytimeDefault::Daytime::MIDNIGHT));
    module->AddObject("UTC"         , PyTimeZone::create(cron::UTC));

    std::cerr << "initializing dtypes\n";
    init_date_dtype(module);
    std::cerr << "done\n";

    return module.release();
  }
  catch (Exception) {
    return nullptr;
  }
}


