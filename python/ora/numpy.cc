#include <cassert>

#include <Python.h>

// Note: Order is important here!
//
// In this, and only this, compilation unit, we need to #include the numpy
// headers without NO_IMPORT_ARRAY #defined.  In all other compilation units,
// this macro is defined, to make sure a single shared copy of the API is used.
// 
// See http://docs.scipy.org/doc/numpy/reference/c-api.array.html#importing-the-api.
//
// FIXME: Encapsulate this so that no human ever ever has to deal with it again.
#define PY_ARRAY_UNIQUE_SYMBOL ora_numpy
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>
#include <numpy/ufuncobject.h>
#include <numpy/npy_3kcompat.h>

#include "py.hh"
#include "np_date.hh"
#include "np_daytime.hh"
#include "numpy.hh"

using namespace py;
using namespace aslib;

//------------------------------------------------------------------------------

namespace {

ref<Object>
date_from_ordinal_date(
  Module* /* module */,
  Tuple* const args,
  Dict* const kw_args)
{
  static char const* arg_names[] = {"year", "ordinal", nullptr};
  PyObject* year_arg;
  PyObject* ordinal_arg;
  PyArray_Descr* dtype = DateDtype<PyDateDefault>::get();
  Arg::ParseTupleAndKeywords(
    args, kw_args, "OO|$O!", arg_names,
    &year_arg, &ordinal_arg, &PyArrayDescr_Type, &dtype);

  // FIXME: Encapsulate this.
  auto const api = (DateDtypeAPI*) dtype->c_metadata;
  assert(api != nullptr);

  return api->function_date_from_ordinal_date(
    Array::FromAny(
      year_arg, aslib::np::YEAR_TYPE, 1, 1, NPY_ARRAY_CARRAY_RO),
    Array::FromAny(
      ordinal_arg, aslib::np::ORDINAL_TYPE, 1, 1, NPY_ARRAY_CARRAY_RO));
}


ref<Object>
date_from_week_date(
  Module* /* module */,
  Tuple* const args,
  Dict* const kw_args)
{
  static char const* arg_names[] 
    = {"week_year", "week", "weekday", "dtype", nullptr};
  PyObject* week_year_arg;
  PyObject* week_arg;
  PyObject* weekday_arg;
  PyArray_Descr* dtype = DateDtype<PyDateDefault>::get();
  Arg::ParseTupleAndKeywords(
    args, kw_args, "OOO|$O!", arg_names,
    &week_year_arg, &week_arg, &weekday_arg, &PyArrayDescr_Type, &dtype);

  // FIXME: Encapsulate this.
  auto const api = (DateDtypeAPI*) dtype->c_metadata;
  assert(api != nullptr);

  return api->function_date_from_week_date(
    Array::FromAny(
      week_year_arg, aslib::np::YEAR_TYPE, 1, 1, NPY_ARRAY_CARRAY_RO),
    Array::FromAny(
      week_arg, aslib::np::WEEK_TYPE, 1, 1, NPY_ARRAY_CARRAY_RO),
    Array::FromAny(
      weekday_arg, aslib::np::WEEKDAY_TYPE, 1, 1, NPY_ARRAY_CARRAY_RO));
}


ref<Object>
date_from_ymd(
  Module* /* module */,
  Tuple* const args,
  Dict* const kw_args)
{
  static char const* arg_names[] = {"year", "month", "day", "dtype", nullptr};
  PyObject* year_arg;
  PyObject* month_arg;
  PyObject* day_arg;
  PyArray_Descr* dtype = DateDtype<PyDateDefault>::get();
  Arg::ParseTupleAndKeywords(
    args, kw_args, "OOO|$O!", arg_names,
    &year_arg, &month_arg, &day_arg, &PyArrayDescr_Type, &dtype);

  // FIXME: Encapsulate this.
  auto const api = (DateDtypeAPI*) dtype->c_metadata;
  assert(api != nullptr);

  return api->function_date_from_ymd(
    Array::FromAny(year_arg, aslib::np::YEAR_TYPE, 1, 1, NPY_ARRAY_CARRAY_RO),
    Array::FromAny(month_arg, aslib::np::MONTH_TYPE, 1, 1, NPY_ARRAY_CARRAY_RO),
    Array::FromAny(day_arg, aslib::np::DAY_TYPE, 1, 1, NPY_ARRAY_CARRAY_RO));
}


ref<Object>
date_from_ymdi(
  Module* /* module */,
  Tuple* const args,
  Dict* const kw_args)
{
  static char const* arg_names[] = {"ymdi", "dtype", nullptr};
  PyObject* ymdi_arg;
  PyArray_Descr* dtype = DateDtype<PyDateDefault>::get();
  Arg::ParseTupleAndKeywords(
    args, kw_args, "O|$O!", arg_names,
    &ymdi_arg, &PyArrayDescr_Type, &dtype);
  auto ymdi_arr
    = Array::FromAny(ymdi_arg, aslib::np::YMDI_TYPE, 1, 1, NPY_ARRAY_CARRAY_RO);

  // OK, we have an aligned 1D int32 array.
  // FIXME: Encapsulate this, and check that it is an ora date dtype.
  auto const api = (DateDtypeAPI*) dtype->c_metadata;
  assert(api != nullptr);

  return api->function_date_from_ymdi(ymdi_arr);
}


// FIXME: Use a 'date' namespace.
auto
functions 
  = Methods<Module>()
    .add<date_from_ordinal_date>    ("date_from_ordinal_date")
    .add<date_from_week_date>       ("date_from_week_date")
    .add<date_from_ymd>             ("date_from_ymd")
    .add<date_from_ymdi>            ("date_from_ymdi")
  ;
  

}  // anonymous namespace

//------------------------------------------------------------------------------

ref<Object>
set_up_numpy(
  Module* const module,
  Tuple* const args,
  Dict* kw_args)
{
  static char const* const arg_names[] = {nullptr};
  Arg::ParseTupleAndKeywords(args, kw_args, "", arg_names);

  // Import numpy stuff.
  if (_import_array() < 0) 
    throw ImportError("failed to import numpy.core.multiarray"); 
  if (_import_umath() < 0) 
    throw ImportError("failed to import numpy.core.umath");

  // Put everything in a submodule `numpy` (even though this is not a package).
  auto const sub = Module::New("ora.ext.numpy");

  DateDtype<PyDate<ora::date::Date>>::add(sub);
  DateDtype<PyDate<ora::date::Date16>>::add(sub);
  DaytimeDtype<PyDaytime<ora::daytime::Daytime>>::add(sub);
  DaytimeDtype<PyDaytime<ora::daytime::Daytime32>>::add(sub);

  sub->AddFunctions(functions);

  sub->AddObject("ORDINAL_DATE_DTYPE",  (Object*) get_ordinal_date_dtype());
  sub->AddObject("WEEK_DATE_DTYPE",     (Object*) get_week_date_dtype());
  sub->AddObject("YMD_DTYPE",           (Object*) get_ymd_dtype());

  module->AddObject("numpy", sub);

  return none_ref();
}


