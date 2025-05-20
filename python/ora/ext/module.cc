#ifdef ORA_NP

// In this, and only this, compilation unit, we need to #include the numpy
// headers without NO_IMPORT_ARRAY #defined.  In all other compilation units,
// this macro is defined, to make sure a single shared copy of the API is used.
// 
// See http://docs.scipy.org/doc/numpy/reference/c-api.array.html#importing-the-api.
//
// FIXME: Encapsulate this so that no human ever ever has to deal with it again.
#define PY_ARRAY_UNIQUE_SYMBOL ora_PyArray_API
#define PY_UFUNC_UNIQUE_SYMBOL ora_PyUFunc_API
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>
#include <numpy/ufuncobject.h>
#include <numpy/npy_3kcompat.h>

#ifdef ORA_BP
# include "np/np.hh"
#endif

#endif

//------------------------------------------------------------------------------

#include <Python.h>
#include <datetime.h>

#include "py_calendar.hh"
#include "py_date_fmt.hh"
#include "py_local.hh"
#include "py_time_fmt.hh"
#include "py_time_zone.hh"
#include "types.hh"

using namespace ora::lib;
using namespace ora::py;

//------------------------------------------------------------------------------

namespace ora {
namespace py {

/* Adds date types.  */
extern void set_up_dates(Module*, Module*);
/* Adds daytime types.  */
extern void set_up_daytimes(Module*, Module*);
/* Adds time types.  */
extern void set_up_times(Module*, Module*);

/* Adds functions from functions.cc.  */
extern Methods<Module>& add_functions(Methods<Module>&);
extern ref<Module> build_np_module();

/* Adds functions from py_cal_functions.cc.  */
extern Methods<Module>& add_cal_functions(Methods<Module>&);

namespace {

Methods<Module> 
methods;


PyModuleDef
module_def{
  PyModuleDef_HEAD_INIT,
  "ora.ext",
  nullptr,
  -1,
  add_cal_functions(add_functions(methods))
};


inline ref<Object>
new_exception(
  Module* const module,
  char const* const name,
  PyObject* const base=PyExc_Exception)
{
  auto const full_name = std::string("ora.") + name;
  auto exc = ora::py::new_exception(full_name.c_str(), base);
  module->AddObject(name, exc);
  return exc;
}


}  // anonymous namespace

}  // namespace py
}  // namespace ora

//------------------------------------------------------------------------------

PyMODINIT_FUNC
PyInit_ext(void)
{
  try {
    auto mod = Module::Create(&module_def);

#ifdef ORA_NP
    // Import numpy.
    // FIXME: Handle if numpy is missing.
    if (_import_array() < 0) 
      throw ImportError("failed to import numpy.core.multiarray"); 
    if (_import_umath() < 0) 
      throw ImportError("failed to import numpy.core.umath");
    bool const np = true;
#endif

    set_up_dates(mod, nullptr);
    set_up_daytimes(mod, nullptr);

    ref<Module> np_mod;
#ifdef ORA_NP
    // FIXME: Move this up, once build_np_module() doesn't require types.
    if (np) {
      np_mod = build_np_module();
      mod->AddObject("np", np_mod);
    }
#endif

    set_up_times(mod, np_mod);

    PyTimeZone  ::add_to(mod, "TimeZone");
    PyLocal     ::add_to(mod, "LocalTime");
    PyCalendar  ::add_to(mod);
    PyDateFmt   ::add_to(mod);
    PyTimeFmt   ::add_to(mod);

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
    mod->AddObject("UTC"              , PyTimeZone::create(ora::UTC));

    TranslateException<ora::InvalidDateError>::to(
      new_exception(mod, "InvalidDateError", PyExc_ValueError));
    TranslateException<ora::DateFormatError>::to(
      new_exception(mod, "DateFormatError", PyExc_ValueError));
    TranslateException<ora::DateRangeError>::to(
      new_exception(mod, "DateRangeError", PyExc_OverflowError));

    TranslateException<ora::InvalidDaytimeError>::to(
      new_exception(mod, "InvalidDaytimeError", PyExc_ValueError));
    TranslateException<ora::DaytimeRangeError>::to(
      new_exception(mod, "DaytimeRangeError", PyExc_OverflowError));

    TranslateException<ora::InvalidTimeError>::to(
      new_exception(mod, "InvalidTimeError", PyExc_ValueError));
    TranslateException<ora::NonexistentDateDaytime>::to(
      new_exception(mod, "NonexistentDateDaytime", PyExc_RuntimeError));
    TranslateException<ora::TimeRangeError>::to(
      new_exception(mod, "TimeRangeError", PyExc_OverflowError));
    TranslateException<ora::TimeFormatError>::to(
      new_exception(mod, "TimeFormatError", PyExc_ValueError));

    TranslateException<ora::CalendarRangeError>::to(
      new_exception(mod, "CalendarRangeError", PyExc_ValueError));

    TranslateException<ora::lib::fs::FileNotFoundError>::to(PyExc_FileNotFoundError);
    TranslateException<ora::lib::RuntimeError>::to(PyExc_RuntimeError);
    TranslateException<FormatError>::to(PyExc_RuntimeError);
    TranslateException<ora::UnknownTimeZoneError>::to(PyExc_ValueError);

    return mod.release();
  }
  catch (Exception) {
    return nullptr;
  }
}


