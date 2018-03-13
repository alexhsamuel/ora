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

#include "np/numpy.hh"

//------------------------------------------------------------------------------

#include <Python.h>
#include <datetime.h>

#include "PyDate.hh"
#include "PyDaytime.hh"
#include "PyLocal.hh"
#include "PyTime.hh"
#include "PyTimeZone.hh"
#include "np/np_time.hh"

using namespace ora::lib;
using namespace ora::py;

//------------------------------------------------------------------------------

namespace ora {
namespace py {

/* Adds functions from functions.cc.  */
extern Methods<Module>& add_functions(Methods<Module>&);
extern ref<Module> build_np_module();

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
};


template<class TIME>
void
add_time(
  char const* name,
  Module* const mod,
  Module* const np_mod)
{
  // If we have numpy, make this type a subtype of numpy.generic.  This is
  // necessary for some numpy operations to work.
  auto const base = np_mod == nullptr ? nullptr : (Type*) &PyGenericArrType_Type;

  Type* type = PyTime<TIME>::set_up("ora."s + name, base);
  mod->AddObject(name, (Object*) type);
  if (np_mod != nullptr)
    TimeDtype<PyTime<TIME>>::set_up(np_mod);
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

    // Import numpy.
    // FIXME: Handle if numpy is missing.
    if (_import_array() < 0) 
      throw ImportError("failed to import numpy.core.multiarray"); 
    if (_import_umath() < 0) 
      throw ImportError("failed to import numpy.core.umath");
    bool const np = true;

    // FIXME: Split up date, time, daytime into separate functions in separate
    // compilation units, as these are where the big template instantiations
    // happen.

    PyDate<ora::date::Date>             ::add_to(mod, "Date");
    PyDate<ora::date::Date16>           ::add_to(mod, "Date16");

    PyDaytime<ora::daytime::Daytime>    ::add_to(mod, "Daytime");
    PyDaytime<ora::daytime::Daytime32>  ::add_to(mod, "Daytime32");
    PyDaytime<ora::daytime::UsecDaytime>::add_to(mod, "UsecDaytime");

    // FIXME: Move this up, once the add_to() above don't do numpy things.
    ref<Module> np_mod;
    if (np) {
      np_mod = build_np_module();
      mod->AddObject("np", np_mod);
    }

    add_time<ora::time::Time>       ("Time"      , mod, np_mod);
    add_time<ora::time::HiTime>     ("HiTime"    , mod, np_mod);
    add_time<ora::time::SmallTime>  ("SmallTime" , mod, np_mod);
    add_time<ora::time::NsTime>     ("NsTime"    , mod, np_mod);
    add_time<ora::time::Unix32Time> ("Unix32Time", mod, np_mod);
    add_time<ora::time::Unix64Time> ("Unix64Time", mod, np_mod);
    add_time<ora::time::Time128>    ("Time128"   , mod, np_mod);

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
    mod->AddObject("UTC"              , PyTimeZone::create(std::make_shared<ora::TimeZone>()));

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


