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
  "cron.ext",
  nullptr,
  -1,
  add_functions(methods)
    .add<set_up_numpy>                ("set_up_numpy")
};


}  // anonymous namespace

//------------------------------------------------------------------------------

PyMODINIT_FUNC
PyInit_ext(void)
{
  auto mod = Module::Create(&module_def);

  try {
    aslib::PyDate<cron::date::Date>             ::add_to(mod, "Date");
    aslib::PyDate<cron::date::Date16>           ::add_to(mod, "Date16");

    aslib::PyDaytime<cron::daytime::Daytime>    ::add_to(mod, "Daytime");
    aslib::PyDaytime<cron::daytime::Daytime32>  ::add_to(mod, "Daytime32");

    aslib::PyTime<cron::time::Time>             ::add_to(mod, "Time");
    aslib::PyTime<cron::time::SmallTime>        ::add_to(mod, "SmallTime");
    aslib::PyTime<cron::time::NsecTime>         ::add_to(mod, "NsecTime");
    aslib::PyTime<cron::time::Unix32Time>       ::add_to(mod, "Unix32Time");
    aslib::PyTime<cron::time::Unix64Time>       ::add_to(mod, "Unix64Time");

    aslib::PyTimeZone                           ::add_to(mod, "TimeZone");

    StructSequenceType* const ymd_date_type = get_ymd_date_type();
    mod->AddObject(ymd_date_type->tp_name, (PyObject*) ymd_date_type);
    StructSequenceType* const hms_daytime_type = get_hms_daytime_type();
    mod->AddObject(hms_daytime_type->tp_name, (PyObject*) hms_daytime_type);

    mod->AddObject("SECOND_INVALID"   , Float::from(cron::SECOND_INVALID));
    mod->AddObject("MINUTE_INVALID"   , Long::from(cron::MINUTE_INVALID));
    mod->AddObject("HOUR_INVALID"     , Long::from(cron::HOUR_INVALID));
    mod->AddObject("DAY_INVALID"      , Long::from(cron::DAY_INVALID));
    mod->AddObject("MONTH_INVALID"    , Long::from(cron::MONTH_INVALID));
    mod->AddObject("YEAR_INVALID"     , Long::from(cron::YEAR_INVALID));
    mod->AddObject("ORDINAL_INVALID"  , Long::from(cron::ORDINAL_INVALID));
    mod->AddObject("WEEK_INVALID"     , Long::from(cron::WEEK_INVALID));
    mod->AddObject("WEEKDAY_INVALID"  , Long::from(cron::WEEKDAY_INVALID));
    mod->AddObject("DAYTICK_INVALID"  , Long::from(cron::DAYTICK_INVALID));
    mod->AddObject("DATENUM_INVALID"  , Long::from(cron::DATENUM_INVALID));
    mod->AddObject("SSM_INVALID"      , Float::from(cron::SSM_INVALID));
    mod->AddObject("DATENUM_MIN"      , Long::from(cron::DATENUM_MIN));
    mod->AddObject("DATENUM_MAX"      , Long::from(cron::DATENUM_MAX));
    mod->AddObject("MIDNIGHT"         , PyDaytimeDefault::create(PyDaytimeDefault::Daytime::MIDNIGHT));
    mod->AddObject("UTC"              , PyTimeZone::create(cron::UTC));

    // FIXME: Use specific Python exception classes.
    TranslateException<cron::InvalidDateError>::to(PyExc_ValueError);
    TranslateException<cron::InvalidDaytimeError>::to(PyExc_ValueError);
    TranslateException<cron::InvalidTimeError>::to(PyExc_ValueError);
    TranslateException<cron::DateFormatError>::to(PyExc_ValueError); 
    TranslateException<cron::DateRangeError>::to(PyExc_OverflowError);
    TranslateException<cron::DaytimeRangeError>::to(PyExc_OverflowError);
    TranslateException<cron::NonexistentLocalTime>::to(PyExc_RuntimeError);
    TranslateException<cron::TimeRangeError>::to(PyExc_OverflowError);

    return mod.release();
  }
  catch (Exception) {
    return nullptr;
  }
}


