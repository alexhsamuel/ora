#include <algorithm>
#include <Python.h>

#include "mem.hh"
#include "py.hh"
#include "numpy.hh"
#include "PyDate.hh"

// FIXME: Check GIL flags.

namespace alxs {

using namespace py;
using namespace py::np;

//------------------------------------------------------------------------------

bool constexpr
PRINT_ARR_FUNCS
  = true;


// Ufunc objects.
UnaryUFunc ufunc_year("year", NPY_INT16);


template<typename PYDATE>
class 
DateDtype
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

  // Ufuncs loop functions.
  static void           loop_year(char**, npy_intp*, npy_intp*, void*);

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
    descr_->c_metadata      = nullptr;
    descr_->hash            = -1;

    if (PyArray_RegisterDataType(descr_) < 0)
      throw py::Exception();
  }

  return descr_;
}


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

  ufunc_year.add_loop(dtype->type_num, loop_year);
  ufunc_year.add_to_module(module);

// FIXME
#if 0
  ref<Object> ufunc = module->GetAttrString("year", false);
  if (ufunc == nullptr) {
    ufunc = ref<Object>::take(PyUFunc_FromFuncAndData(
      nullptr, nullptr, nullptr, 0, 1, 1, 
      PyUFunc_None, "year", "year_docstring", 0));
    module->AddObject("year", ufunc);
  }
  int arg_types[] = {dtype->type_num, NPY_INT16};
  check_zero(PyUFunc_RegisterLoopForType(
   (PyUFuncObject*) (PyObject*) ufunc, 
   dtype->type_num, ufunc_year, arg_types, nullptr));
#endif
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
void 
DateDtype<PYDATE>::loop_year(
  char** const args, 
  npy_intp* const dimensions,
  npy_intp* const steps, 
  void* const /* data */)
{
  auto const n          = dimensions[0];
  auto const ar0_step   = steps[0];
  auto const res_step   = steps[1];
  auto ar0              = (Date const*) args[0];
  auto res              = (uint16_t*) args[1];

  for (npy_intp i = 0; i < n; i++) {
    *res = ar0->get_parts().year;
    ar0 = step(ar0, ar0_step);
    res = step(res, res_step);
  }
}


//------------------------------------------------------------------------------

template<typename PYDATE>
PyArray_Descr*
DateDtype<PYDATE>::descr_
  = nullptr;


//------------------------------------------------------------------------------

}  // namespace alxs


