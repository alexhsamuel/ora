#include "numpy.hh"
#include "np_types.hh"

namespace aslib {

//------------------------------------------------------------------------------

PyArray_Descr*
get_ordinal_date_dtype()
{
  static PyArray_Descr* dtype = nullptr;
  if (dtype == nullptr) {
    // Lazy one-time initialization.
    auto const fields = take_not_null<Object>(Py_BuildValue(
      "[(ss)(ss)]", "year", "<i2", "ordinal", "<u2"));
    auto rval = PyArray_DescrConverter(fields, &dtype);
    check_one(rval);
    assert(dtype != nullptr);
  }

  return dtype;
}


PyArray_Descr*
get_week_date_dtype()
{
  static PyArray_Descr* dtype = nullptr;
  if (dtype == nullptr) {
    // Lazy one-time initialization.
    auto const fields = take_not_null<Object>(Py_BuildValue(
      "[(ss)(ss)(ss)]", "week_year", "<i2", "week", "u1", "weekday", "u1"));
    auto rval = PyArray_DescrConverter(fields, &dtype);
    check_one(rval);
    assert(dtype != nullptr);
  }

  return dtype;
}


PyArray_Descr*
get_ymd_dtype()
{
  static PyArray_Descr* dtype = nullptr;
  if (dtype == nullptr) {
    // Lazy one-time initialization.
    auto const fields = take_not_null<Object>(Py_BuildValue(
      "[(ss)(ss)(ss)]", "year", "<i2", "month", "u1", "day", "u1"));
    auto rval = PyArray_DescrConverter(fields, &dtype);
    check_one(rval);
    assert(dtype != nullptr);
  }

  return dtype;
}


//------------------------------------------------------------------------------

}  // namespace aslib

