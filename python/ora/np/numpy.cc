#include <cassert>

#include <Python.h>

#include "py.hh"
#include "np_date.hh"
#include "np_daytime.hh"
#include "np_time.hh"
#include "numpy.hh"

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
  static char const* arg_names[] = {"year", "ordinal", "Date", nullptr};
  PyObject* year_arg;
  PyObject* ordinal_arg;
  PyArray_Descr* descr = DateDtype<PyDateDefault>::get();
  Arg::ParseTupleAndKeywords(
    args, kw_args, "OO|$O&", arg_names,
    &year_arg, &ordinal_arg, &PyArray_DescrConverter2, &descr);
  if (descr == nullptr)
    throw TypeError("not an ora date dtype");
  auto const api = DateAPI::from(descr);

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
  auto const api = (DateAPI*) dtype->c_metadata;
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
  auto const api = (DateAPI*) dtype->c_metadata;
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
  auto const api = (DateAPI*) dtype->c_metadata;
  assert(api != nullptr);

  return api->function_date_from_ymdi(ymdi_arr);
}


ref<Object>
from_offset(
  Module*,
  Tuple* const args,
  Dict* const kw_args)
{
  static char const* arg_names[] = {"offset", "dtype", nullptr};
  PyObject* offset_arg;
  Descr* dtype = TimeDtype<PyTimeDefault>::get_descr();
  // FIXME: DescrConverter2 and check for failure.
  Arg::ParseTupleAndKeywords(
    args, kw_args, "O|$O&", arg_names,
    &offset_arg, &PyArray_DescrConverter, &dtype);
  auto offset = Array::FromAny(offset_arg, NPY_INT64, 0, 0, NPY_ARRAY_BEHAVED);

  // FIXME: Handle dtype == nullptr in TimeAPI::get().
  return TimeAPI::from(dtype)->from_offset(offset);
}


ref<Object>
from_local(
  Module*,
  Tuple* const args,
  Dict* const kw_args)
{
  // FIXME: Accept a 'first' argument.
  static char const* arg_names[] 
    = {"date", "daytime", "time_zone", "Time", nullptr};
  Object* date_arg;
  Object* daytime_arg;
  Object* tz_arg;
  bool first = true;
  PyArray_Descr* time_descr = TimeDtype<PyTimeDefault>::get_descr();
  Arg::ParseTupleAndKeywords(
    args, kw_args, "OOO|$O&", arg_names, 
    &date_arg, &daytime_arg, &tz_arg, 
    &PyArray_DescrConverter2, &time_descr
    );

  if (time_descr == nullptr)
    throw TypeError("not an ora time dtype");

  auto const date_arr       = to_date_array(date_arg);
  auto const date_descr     = date_arr->descr();
  auto const date_api       = DateAPI::from(date_descr);

  auto const daytime_arr    = to_daytime_array(daytime_arg);
  auto const daytime_descr  = daytime_arr->descr();
  auto const daytime_api    = DaytimeAPI::from(daytime_descr);

  auto const tz             = convert_to_time_zone(tz_arg);

  // Get the time dtype API for the time type.  This also confirms the dtype
  // is a time dtype.
  auto const time_api       = TimeAPI::from(time_descr);

  size_t constexpr nargs = 3;
  PyArrayObject* op[nargs] = {
    (PyArrayObject*) (Array*) date_arr, 
    (PyArrayObject*) (Array*) daytime_arr,
    nullptr,
  };
  npy_uint32 flags[nargs] = {
    NPY_ITER_READONLY, 
    NPY_ITER_READONLY,
    NPY_ITER_WRITEONLY | NPY_ITER_ALLOCATE,
  };
  PyArray_Descr* dtypes[nargs] = {date_descr, daytime_descr, time_descr};

  // Construct the iterator.  We'll handle the inner loop explicitly.
  auto const iter = NpyIter_MultiNew(
    nargs, op, NPY_ITER_EXTERNAL_LOOP, NPY_KEEPORDER, NPY_UNSAFE_CASTING, 
    flags, dtypes);
  if (iter == nullptr)
    throw Exception();

  auto const next           = NpyIter_GetIterNext(iter, nullptr);
  auto const& inner_stride  = NpyIter_GetInnerStrideArray(iter);

  auto const& inner_size    = *NpyIter_GetInnerLoopSizePtr(iter);
  auto const data_ptrs      = NpyIter_GetDataPtrArray(iter);

  do {
    auto d_ptr = data_ptrs[0];
    auto y_ptr = data_ptrs[1];
    auto t_ptr = data_ptrs[2];

    auto const d_stride = inner_stride[0];
    auto const y_stride = inner_stride[1];
    auto const t_stride = inner_stride[2];

    for (auto size = inner_size;
         size > 0;
         --size,
         d_ptr += d_stride,
         y_ptr += y_stride,
         t_ptr += t_stride) {
      auto const datenum = date_api->get_datenum(d_ptr);
      auto const daytick = daytime_api->get_daytick(y_ptr);
      time_api->from_local(datenum, daytick, *tz, first, t_ptr);
    }
  } while (next(iter));

  // Get the result from the iterator object array.
  auto ret = ref<Array>::of((Array*) NpyIter_GetOperandArray(iter)[2]);
  check_succeed(NpyIter_Deallocate(iter));
  return std::move(ret);
}


ref<Object>
to_local(
  Module*,
  Tuple* const args,
  Dict* const kw_args)
{
  static char const* arg_names[] 
    = {"time", "time_zone", "Date", "Daytime", nullptr};
  Object* time_arg;
  Object* tz_arg;
  PyArray_Descr* date_descr = DateDtype<PyDateDefault>::get();
  PyArray_Descr* daytime_descr = DaytimeDtype<PyDaytimeDefault>::get();
  Arg::ParseTupleAndKeywords(
    args, kw_args, "OO|$O&O&", arg_names, 
    &time_arg, &tz_arg, 
    &PyArray_DescrConverter2, &date_descr,
    &PyArray_DescrConverter2, &daytime_descr
    );

  if (date_descr == nullptr)
    throw TypeError("not an ora date dtype");
  if (daytime_descr == nullptr)
    throw TypeError("not an ora daytime dtype");

  auto const time_arr   = to_time_array(time_arg);
  auto const time_descr = time_arr->descr();
  auto const time_api   = TimeAPI::from(time_descr);

  auto const tz = convert_to_time_zone(tz_arg);

  // Get the date dtype API for this date type.  This also confirms the dtype
  // is a date dtype.
  auto const date_api = DateAPI::from(date_descr);
  // Allocate the output date array.
  auto date_arr = Array::NewLikeArray(time_arr, NPY_CORDER, date_descr);

  // Get the daytime dtype API for this daytime type.  This also confirms the
  // dtype is a daytime dtype.
  auto const daytime_api = DaytimeAPI::from(daytime_descr);
  // Allocate the output daytime array.
  auto daytime_arr = Array::NewLikeArray(time_arr, NPY_CORDER, daytime_descr);


  size_t constexpr nargs = 3;
  PyArrayObject* op[nargs] = {
    (PyArrayObject*) (Array*) time_arr, 
    (PyArrayObject*) (Array*) date_arr, 
    (PyArrayObject*) (Array*) daytime_arr,
  };
  npy_uint32 flags[nargs] = {
    NPY_ITER_READONLY, 
    NPY_ITER_WRITEONLY,
    NPY_ITER_WRITEONLY,
  };
  PyArray_Descr* dtypes[nargs] = {time_descr, date_descr, daytime_descr};

  // Construct the iterator.  We'll handle the inner loop explicitly.
  auto const iter = NpyIter_MultiNew(
    nargs, op, NPY_ITER_EXTERNAL_LOOP, NPY_KEEPORDER, NPY_UNSAFE_CASTING, 
    flags, dtypes);
  if (iter == nullptr)
    throw Exception();

  auto const next           = NpyIter_GetIterNext(iter, nullptr);
  auto const& inner_stride  = NpyIter_GetInnerStrideArray(iter);

  auto const& inner_size    = *NpyIter_GetInnerLoopSizePtr(iter);
  auto const data_ptrs      = NpyIter_GetDataPtrArray(iter);

  do {
    auto t_ptr = data_ptrs[0];
    auto d_ptr = data_ptrs[1];
    auto y_ptr = data_ptrs[2];

    auto const t_stride = inner_stride[0];
    auto const d_stride = inner_stride[1];
    auto const y_stride = inner_stride[2];

    for (auto size = inner_size;
         size > 0;
         --size,
         t_ptr += t_stride,
         d_ptr += d_stride,
         y_ptr += y_stride) {
      auto ldd = time_api->to_local_datenum_daytick(t_ptr, *tz);
      // Convert to date, daytime.  If the date is invalid, make the daytime
      // invalid too.
      if (!date_api->from_datenum(ldd.datenum, d_ptr))
        ldd.daytick = ora::DAYTICK_INVALID;
      daytime_api->from_daytick(ldd.daytick, y_ptr);
    }
  } while (next(iter));

  check_succeed(NpyIter_Deallocate(iter));

  return Tuple::builder << std::move(date_arr) << std::move(daytime_arr);
}


auto
functions 
  = Methods<Module>()
    .add<date_from_ordinal_date>    ("date_from_ordinal_date")
    .add<date_from_week_date>       ("date_from_week_date")
    .add<date_from_ymd>             ("date_from_ymd")
    .add<date_from_ymdi>            ("date_from_ymdi")
    .add<from_local>                ("from_local")
    .add<from_offset>               ("from_offset")
    .add<to_local>                  ("to_local")
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
  DaytimeDtype<PyDaytime<ora::daytime::UsecDaytime>>::add(mod);

  mod->AddFunctions(functions);

  mod->AddObject("ORDINAL_DATE_DTYPE",  (Object*) get_ordinal_date_dtype());
  mod->AddObject("WEEK_DATE_DTYPE",     (Object*) get_week_date_dtype());
  mod->AddObject("YMD_DTYPE",           (Object*) get_ymd_dtype());

  return mod;
}


}  // namespace py
}  // namespace ora

