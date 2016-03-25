#include <Python.h>
#include <datetime.h>

#include "PyDate.hh"
#include "PyDaytime.hh"
#include "PyTime.hh"
#include "PyTimeZone.hh"

using namespace aslib;
using namespace py;

//------------------------------------------------------------------------------

/* Adds functions from functions.cc.  */
extern Methods<Module>& add_functions(Methods<Module>&);

/* The numpy setup function in numpy.cc  */
extern ref<Object> set_up_numpy(Module*, Tuple*, Dict*);

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
    .add<set_up_numpy>                ("set_up_numpy")
};


}  // anonymous namespace

//------------------------------------------------------------------------------

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
    aslib::PyDate<cron::Date>            ::add_to(module, "Date");
    aslib::PyDate<cron::Date16>          ::add_to(module, "Date16");

    aslib::PyDaytime<cron::Daytime>      ::add_to(module, "Daytime");
    aslib::PyDaytime<cron::SmallDaytime> ::add_to(module, "SmallDaytime");

    aslib::PyTime<cron::Time>            ::add_to(module, "Time");
    aslib::PyTime<cron::SmallTime>       ::add_to(module, "SmallTime");
    aslib::PyTime<cron::NsecTime>        ::add_to(module, "NsecTime");
    aslib::PyTime<cron::Unix32Time>      ::add_to(module, "Unix32Time");
    aslib::PyTime<cron::Unix64Time>      ::add_to(module, "Unix64Time");

    aslib::PyTimeZone                    ::add_to(module, "TimeZone");

    StructSequenceType* const parts_type = get_date_parts_type();
    module->AddObject(parts_type->tp_name, (PyObject*) parts_type);

    module->AddObject("DATENUM_MIN" , Long::FromLong(cron::DATENUM_MIN));
    module->AddObject("DATENUM_MAX" , Long::FromLong(cron::DATENUM_MAX));
    module->AddObject("MIDNIGHT"    , PyDaytimeDefault::create(PyDaytimeDefault::Daytime::MIDNIGHT));
    module->AddObject("UTC"         , PyTimeZone::create(cron::UTC));

    TranslateException<cron::InvalidDateError>::to(PyExc_ValueError);
    TranslateException<cron::DateRangeError>::to(PyExc_OverflowError);

    return module.release();
  }
  catch (Exception) {
    return nullptr;
  }
}


