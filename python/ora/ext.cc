#include <Python.h>
#include <datetime.h>

#include "PyDate.hh"
#include "PyDaytime.hh"
#include "PyLocal.hh"
#include "PyTime.hh"
#include "PyTimeZone.hh"

using namespace ora::lib;
using namespace ora::py;

//------------------------------------------------------------------------------

namespace ora {
namespace py {

/* Adds functions from functions.cc.  */
extern Methods<Module>& add_functions(Methods<Module>&);

#ifdef ORA_NUMPY
/* The numpy setup function in numpy.cc  */
extern ref<Object> set_up_numpy(Module*, Tuple*, Dict*);
#endif

namespace {

Methods<Module> 
methods;


PyModuleDef
module_def{
  PyModuleDef_HEAD_INIT,
  "ora.ext",
  nullptr,
  -1,
  add_functions(methods)
#ifdef ORA_NUMPY
    .add<set_up_numpy>                ("set_up_numpy")
#endif
};


}  // anonymous namespace

}  // namespace py
}  // namespace ora

//------------------------------------------------------------------------------

PyMODINIT_FUNC
PyInit_ext(void)
{
  auto mod = Module::Create(&module_def);

  try {
    PyDate<ora::date::Date>             ::add_to(mod, "Date");
    PyDate<ora::date::Date16>           ::add_to(mod, "Date16");

    PyDaytime<ora::daytime::Daytime>    ::add_to(mod, "Daytime");
    PyDaytime<ora::daytime::Daytime32>  ::add_to(mod, "Daytime32");
    PyDaytime<ora::daytime::UsecDaytime>::add_to(mod, "UsecDaytime");

    PyTime<ora::time::Time>             ::add_to(mod, "Time");
    PyTime<ora::time::HiTime>           ::add_to(mod, "HiTime");
    PyTime<ora::time::SmallTime>        ::add_to(mod, "SmallTime");
    PyTime<ora::time::NsTime>           ::add_to(mod, "NsTime");
    PyTime<ora::time::Unix32Time>       ::add_to(mod, "Unix32Time");
    PyTime<ora::time::Unix64Time>       ::add_to(mod, "Unix64Time");
    PyTime<ora::time::Time128>          ::add_to(mod, "Time128");

    PyTimeZone                          ::add_to(mod, "TimeZone");
    PyLocal                             ::add_to(mod, "Local");

    StructSequenceType* const ymd_date_type = get_ymd_date_type();
    mod->AddObject(ymd_date_type->tp_name, (PyObject*) ymd_date_type);
    StructSequenceType* const hms_daytime_type = get_hms_daytime_type();
    mod->AddObject(hms_daytime_type->tp_name, (PyObject*) hms_daytime_type);

    mod->AddObject("SECOND_INVALID"   , Float::from(ora::SECOND_INVALID));
    mod->AddObject("MINUTE_INVALID"   , Long::from(ora::MINUTE_INVALID));
    mod->AddObject("HOUR_INVALID"     , Long::from(ora::HOUR_INVALID));
    mod->AddObject("DAY_INVALID"      , Long::from(ora::DAY_INVALID));
    mod->AddObject("MONTH_INVALID"    , Long::from(ora::MONTH_INVALID));
    mod->AddObject("YEAR_INVALID"     , Long::from(ora::YEAR_INVALID));
    mod->AddObject("ORDINAL_INVALID"  , Long::from(ora::ORDINAL_INVALID));
    mod->AddObject("WEEK_INVALID"     , Long::from(ora::WEEK_INVALID));
    mod->AddObject("WEEKDAY_INVALID"  , Long::from(ora::WEEKDAY_INVALID));
    mod->AddObject("DAYTICK_INVALID"  , Long::from(ora::DAYTICK_INVALID));
    mod->AddObject("DATENUM_INVALID"  , Long::from(ora::DATENUM_INVALID));
    mod->AddObject("SSM_INVALID"      , Float::from(ora::SSM_INVALID));
    mod->AddObject("DATENUM_MIN"      , Long::from(ora::DATENUM_MIN));
    mod->AddObject("DATENUM_MAX"      , Long::from(ora::DATENUM_MAX));
    mod->AddObject("MIDNIGHT"         , PyDaytimeDefault::create(PyDaytimeDefault::Daytime::MIDNIGHT));
    mod->AddObject("UTC"              , 
                   PyTimeZone::create(std::make_shared<ora::TimeZone>()));

    // FIXME: Use specific Python exception classes.
    TranslateException<ora::InvalidDateError>::to(PyExc_ValueError);
    TranslateException<ora::InvalidDaytimeError>::to(PyExc_ValueError);
    TranslateException<ora::InvalidTimeError>::to(PyExc_ValueError);
    TranslateException<ora::DateFormatError>::to(PyExc_ValueError); 
    TranslateException<ora::DateRangeError>::to(PyExc_OverflowError);
    TranslateException<ora::DaytimeRangeError>::to(PyExc_OverflowError);
    TranslateException<ora::NonexistentDateDaytime>::to(PyExc_RuntimeError);
    TranslateException<ora::TimeRangeError>::to(PyExc_OverflowError);
    TranslateException<ora::TimeFormatError>::to(PyExc_ValueError);
    TranslateException<ora::lib::fs::FileNotFoundError>::to(PyExc_FileNotFoundError);
    TranslateException<ora::lib::RuntimeError>::to(PyExc_RuntimeError);
    TranslateException<FormatError>::to(PyExc_RuntimeError);

    return mod.release();
  }
  catch (Exception) {
    return nullptr;
  }
}


