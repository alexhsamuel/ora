#include <algorithm>
#include <Python.h>

#include "aslib/mem.hh"
#include "cron/date_functions.hh"
#include "py.hh"
#include "np_types.hh"
#include "numpy.hh"
#include "PyDate.hh"

// FIXME: Check GIL flags.

namespace aslib {

using namespace py;
using namespace py::np;

//------------------------------------------------------------------------------

bool constexpr
PRINT_ARR_FUNCS
  = true;


class DateDtypeAPI
{
public:

  virtual ref<Object> function_date_from_ymdi(Array*) = 0;

};


template<typename PYDATE>
class DateDtype
{
public:

  using Date = typename PYDATE::Date;

  /*
   * Returns the singleton descriptor / dtype object.
   */
  static PyArray_Descr* get();

  /*
   * Adds the dtype object to the Python type object as the `dtype` attribute.
   */
  static void           add(Module*);

private:

  // FIXME: Wrap these.
  static void           copyswap(Date*, Date const*, int, PyArrayObject*);
  static void           copyswapn(Date*, npy_intp, Date const*, npy_intp, npy_intp, int, PyArrayObject*);
  static Object*        getitem(Date const*, PyArrayObject*);
  static int            setitem(Object*, Date*, PyArrayObject*);

  class API
  : public DateDtypeAPI
  {
  public:

    virtual ref<Object> function_date_from_ymdi(Array*);

  };

  static PyArray_Descr* descr_;

};


template<typename PYDATE>
PyArray_Descr*
DateDtype<PYDATE>::get()
{
  if (descr_ == nullptr) {
    // Deliberately 'leak' this instance, as it has process lifetime.
    auto const arr_funcs = new PyArray_ArrFuncs;
    PyArray_InitArrFuncs(arr_funcs);
    arr_funcs->copyswap         = (PyArray_CopySwapFunc*) copyswap;
    arr_funcs->copyswapn        = (PyArray_CopySwapNFunc*) copyswapn;
    arr_funcs->getitem          = (PyArray_GetItemFunc*) getitem;
    arr_funcs->setitem          = (PyArray_SetItemFunc*) setitem;

    descr_ = PyObject_New(PyArray_Descr, &PyArrayDescr_Type);
    descr_->typeobj         = incref(&PYDATE::type_);
    descr_->kind            = 'V';
    descr_->type            = 'j';
    descr_->byteorder       = '=';
    descr_->flags           = 0;
    descr_->type_num        = 0;
    descr_->elsize          = sizeof(Date);
    descr_->alignment       = alignof(Date);
    descr_->subarray        = nullptr;
    descr_->fields          = nullptr;
    descr_->names           = nullptr;
    descr_->f               = arr_funcs;
    descr_->metadata        = nullptr;
    descr_->c_metadata      = (NpyAuxData*) new API();
    descr_->hash            = -1;

    if (PyArray_RegisterDataType(descr_) < 0)
      throw py::Exception();
  }

  return descr_;
}


// FIXME: Remove this once Month, Day are 1-indexed.
namespace {

template<typename DATE>
inline cron::YmdDate
get_ymd_(
  DATE const date)
{
  if (date.is_valid()) {
    cron::YmdDate ymd = date.get_ymd();
    ymd.month++;
    ymd.day++;
    return ymd;
  }
  else
    return cron::YmdDate::get_invalid();
}


}  // anonymous namespace


template<typename PYDATE>
void
DateDtype<PYDATE>::add(
  Module* const module)
{
  // Build or get the dtype.
  auto const dtype = DateDtype<PYDATE>::get();

  // Add the dtype as a class attribute.
  auto const dict = (Dict*) dtype->typeobj->tp_dict;
  assert(dict != nullptr);
  dict->SetItemString("dtype", (Object*) dtype);

  create_or_get_ufunc(module, "get_day", 1, 1)->add_loop_1(
    dtype->type_num, NPY_UINT8, 
    ufunc_loop_1<Date, uint8_t, cron::get_day<Date>>);
  
  create_or_get_ufunc(module, "get_month", 1, 1)->add_loop_1(
    dtype->type_num, NPY_UINT8, 
    ufunc_loop_1<Date, uint8_t, cron::get_month<Date>>);

  create_or_get_ufunc(module, "get_year", 1, 1)->add_loop_1(
    dtype->type_num, NPY_INT16, 
    ufunc_loop_1<Date, int16_t, cron::get_year<Date>>);

  create_or_get_ufunc(module, "get_ymd", 1, 1)->add_loop_1(
    dtype, get_ymd_dtype(),
    ufunc_loop_1<Date, cron::YmdDate, cron::get_ymd_<Date>>);

  create_or_get_ufunc(module, "get_ymdi", 1, 1)->add_loop_1(
    dtype->type_num, NPY_INT32, 
    ufunc_loop_1<Date, int32_t, cron::get_ymdi<Date>>);
}


//------------------------------------------------------------------------------
// numpy array functions

template<typename PYDATE>
void
DateDtype<PYDATE>::copyswap(
  Date* const dst,
  Date const* const src,
  int const swap,
  PyArrayObject* const arr)
{
  if (PRINT_ARR_FUNCS)
    std::cerr << "copyswap\n";
  if (swap)
    copy_swapped<sizeof(Date)>(src, dst);
  else
    copy<sizeof(Date)>(src, dst);
}


template<typename PYDATE>
void 
DateDtype<PYDATE>::copyswapn(
  Date* const dst, 
  npy_intp const dst_stride, 
  Date const* const src, 
  npy_intp const src_stride, 
  npy_intp const n, 
  int const swap, 
  PyArrayObject* const arr)
{
  if (PRINT_ARR_FUNCS)
    std::cerr << "copyswapn(" << n << ")\n";
  if (src_stride == 0) {
    // Swapper or unswapped fill.  Optimize this special case.
    Date date;
    if (swap) 
      copy_swapped<sizeof(Date)>(src, &date);
    else
      date = *src;
    Date* d = dst;
    for (npy_intp i = 0; i < n; ++i) {
      *d = date;
      d = (Date*) (((char*) d) + dst_stride);
    }
  }
  else {
    char const* s = (char const*) src;
    char* d = (char*) dst;
    if (swap) 
      for (npy_intp i = 0; i < n; ++i) {
        copy_swapped<sizeof(Date)>(s, d);
        s += src_stride;
        d += dst_stride;
      }
    else 
      for (npy_intp i = 0; i < n; ++i) {
        copy<sizeof(Date)>(s, d);
        s += src_stride;
        d += dst_stride;
      }
  }
}


template<typename PYDATE>
Object*
DateDtype<PYDATE>::getitem(
  Date const* const data,
  PyArrayObject* const arr)
{
  if (PRINT_ARR_FUNCS)
    std::cerr << "getitem\n";
  return PYDATE::create(*data).release();
}


template<typename PYDATE>
int
DateDtype<PYDATE>::setitem(
  Object* const item,
  Date* const data,
  PyArrayObject* const arr)
{
  if (PRINT_ARR_FUNCS)
    std::cerr << "setitem\n";
  try {
    *data = convert_to_date<Date>(item);
  }
  catch (Exception) {
    return -1;
  }
  return 0;
}


//------------------------------------------------------------------------------

template<typename PYDATE>
ref<Object>
DateDtype<PYDATE>::API::function_date_from_ymdi(
  Array* const ymdi_arr)
{
  using Date = typename PYDATE::Date;

  // Create the output array.
  auto const size = ymdi_arr->size();
  auto date_arr = Array::SimpleNew1D(size, descr_->type_num);
  // Fill it.
  auto const y = ymdi_arr->get_const_ptr<int>();
  auto const d = date_arr->get_ptr<Date>();
  for (npy_intp i = 0; i < size; ++i)
    d[i] = cron::from_ymdi<Date>(y[i]);

  return std::move(date_arr);
}


//------------------------------------------------------------------------------

template<typename PYDATE>
PyArray_Descr*
DateDtype<PYDATE>::descr_
  = nullptr;


//------------------------------------------------------------------------------

}  // namespace aslib

