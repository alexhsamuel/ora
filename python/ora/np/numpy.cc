#include <cassert>

#include <Python.h>

#include "py.hh"
#include "np_date.hh"
#include "np_daytime.hh"
#include "np_time.hh"
#include "numpy.hh"
#include "PyTime.hh"

using namespace ora::lib;
using namespace ora::py;

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
      year_arg, np::YEAR_TYPE, 1, 1, NPY_ARRAY_CARRAY_RO),
    Array::FromAny(
      ordinal_arg, np::ORDINAL_TYPE, 1, 1, NPY_ARRAY_CARRAY_RO));
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
      week_year_arg, np::YEAR_TYPE, 1, 1, NPY_ARRAY_CARRAY_RO),
    Array::FromAny(
      week_arg, np::WEEK_TYPE, 1, 1, NPY_ARRAY_CARRAY_RO),
    Array::FromAny(
      weekday_arg, np::WEEKDAY_TYPE, 1, 1, NPY_ARRAY_CARRAY_RO));
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
    Array::FromAny(year_arg, np::YEAR_TYPE, 1, 1, NPY_ARRAY_CARRAY_RO),
    Array::FromAny(month_arg, np::MONTH_TYPE, 1, 1, NPY_ARRAY_CARRAY_RO),
    Array::FromAny(day_arg, np::DAY_TYPE, 1, 1, NPY_ARRAY_CARRAY_RO));
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
    = Array::FromAny(ymdi_arg, np::YMDI_TYPE, 1, 1, NPY_ARRAY_CARRAY_RO);

  // OK, we have an aligned 1D int32 array.
  // FIXME: Encapsulate this, and check that it is an ora date dtype.
  auto const api = (DateDtypeAPI*) dtype->c_metadata;
  assert(api != nullptr);

  return api->function_date_from_ymdi(ymdi_arr);
}


// FIXME: Put in a time source file.
ref<Object>
from_offset(
  Module*,
  Tuple* const args,
  Dict* const kw_args)
{
  static char const* arg_names[] = {"offset", "dtype", nullptr};
  PyObject* offset_arg;
  Descr* dtype;
  Arg::ParseTupleAndKeywords(
    args, kw_args, "O|$O&", arg_names,
    &offset_arg, &PyArray_DescrConverter, &dtype);
  auto offset = Array::FromAny(offset_arg, NPY_INT64, 0, 0, NPY_ARRAY_BEHAVED);

  return TimeAPI::get(dtype)->from_offset(offset);
}


// FIXME: Use a 'date' namespace.
auto
functions 
  = Methods<Module>()
    .add<date_from_ordinal_date>    ("date_from_ordinal_date")
    .add<date_from_week_date>       ("date_from_week_date")
    .add<date_from_ymd>             ("date_from_ymd")
    .add<date_from_ymdi>            ("date_from_ymdi")
    .add<from_offset>               ("from_offset")
  ;
  

}  // anonymous namespace

//------------------------------------------------------------------------------

namespace ora {
namespace py {

ref<Module>
build_np_module()
{
  // Put everything in a submodule `np` (even though this is not a package).
  auto mod = Module::New("ora.ext.np");

  DateDtype<PyDate<ora::date::Date>>::add(mod);
  DateDtype<PyDate<ora::date::Date16>>::add(mod);
  DaytimeDtype<PyDaytime<ora::daytime::Daytime>>::add(mod);
  DaytimeDtype<PyDaytime<ora::daytime::Daytime32>>::add(mod);

  mod->AddFunctions(functions);

  mod->AddObject("ORDINAL_DATE_DTYPE",  (Object*) get_ordinal_date_dtype());
  mod->AddObject("WEEK_DATE_DTYPE",     (Object*) get_week_date_dtype());
  mod->AddObject("YMD_DTYPE",           (Object*) get_ymd_dtype());

  return mod;
}


}  // namespace py
}  // namespace ora

