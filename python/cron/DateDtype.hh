#include <algorithm>
#include <Python.h>

#include "py.hh"
#include "PyDate.hh"

namespace alxs {

using namespace py;

//------------------------------------------------------------------------------

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
   * Adds the dtype object to the Python type object as the 'dtype' attribute.
   */
  static void           add();

private:

  // FIXME: Wrap these.
  static void           copyswap(Date*, Date const*, int, PyArrayObject*);
  static void           fill(Date*, npy_intp, PyArrayObject*);
  static void           fillwithscalar(Date*, npy_intp, Date const*, PyArrayObject*);
  static Object*        getitem(Date const*, PyArrayObject*);
  static int            setitem(Object*, Date*, PyArrayObject*);

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
    arr_funcs->fill             = (PyArray_FillFunc*) fill;
    arr_funcs->fillwithscalar   = (PyArray_FillWithScalarFunc*) fillwithscalar;
    arr_funcs->getitem          = (PyArray_GetItemFunc*) getitem;
    arr_funcs->setitem          = (PyArray_SetItemFunc*) setitem;

    descr_ = PyObject_New(PyArray_Descr, &PyArrayDescr_Type);
    descr_->typeobj         = incref(&PYDATE::type_);
    descr_->kind            = 'V';
    descr_->type            = 'j';
    descr_->byteorder       = '=';
// FIXME
//  descr_->hasobject       = NPY_NEEDS_PYAPI | NPY_USE_GETITEM | NPY_USE_SETITEM;
    descr_->type_num        = 0;
    descr_->elsize          = sizeof(Date);
    descr_->alignment       = alignof(Date);
    descr_->subarray        = nullptr;
    descr_->fields          = nullptr;
    descr_->names           = nullptr;
    descr_->f               = arr_funcs;

    if (PyArray_RegisterDataType(descr_) < 0)
      throw py::Exception();
  }

  return descr_;
}


template<typename PYDATE>
void
DateDtype<PYDATE>::add()
{
  // Build or get the dtype.
  auto const dtype = DateDtype<PYDATE>::get();

  // Add the dtype as a class attribute.
  // FIXME
  std::cerr << "adding dtype\n";

  // auto const dict = (Mapping*) dtype->typeobj->tp_dict;
  // assert(dict != nullptr);
  // dict->SetItemString("dtype", (Object*) dtype);

  // ((Object*) dtype->typeobj)->SetAttrString("dtype", (Object*) dtype);

  // std::cerr << "adding dtype done\n";
}


//------------------------------------------------------------------------------
// numpy array functions

template<typename PYDATE>
void
DateDtype<PYDATE>::copyswap(
  Date* const dest,
  Date const* const src,
  int const swap,
  PyArrayObject* const arr)
{
  std::cerr << "copyswap\n";
  std::cerr << "PyArray_ISBEHAVED_RO = " << PyArray_ISBEHAVED_RO(arr) << "\n";
  assert(!swap);  // FIXME
  *dest = *src;
}


template<typename PYDATE>
void 
DateDtype<PYDATE>::fill(
  Date* const data, 
  npy_intp const length, 
  PyArrayObject* arr)
{
  std::cerr << "fill\n";
  std::cerr << "PyArray_ISBEHAVED_RO = " << PyArray_ISBEHAVED_RO(arr) << "\n";
  assert(length > 1);
  auto const offset = data[1] - data[0];
  auto date = data[1];
  for (npy_intp i = 2; i < length; ++i)
    data[i] = date += offset;
}


template<typename PYDATE>
void 
DateDtype<PYDATE>::fillwithscalar(
  Date* const buffer, 
  npy_intp const length, 
  Date const* const value, 
  PyArrayObject* const arr)
{
  std::cerr << "fillwithscalar\n";
  std::cerr << "PyArray_ISBEHAVED_RO = " << PyArray_ISBEHAVED_RO(arr) << "\n";
  std::fill_n(buffer, length, *value);
}


template<typename PYDATE>
Object*
DateDtype<PYDATE>::getitem(
  Date const* const data,
  PyArrayObject* const arr)
{
  std::cerr << "getitem[" << data - (Date const*) PyArray_DATA(arr) << "]\n";
  std::cerr << "PyArray_ISBEHAVED_RO = " << PyArray_ISBEHAVED_RO(arr) << "\n";
  // FIXME: Check PyArray_ISBEHAVED_RO(arr)?
  return PYDATE::create(*data).release();
}


template<typename PYDATE>
int
DateDtype<PYDATE>::setitem(
  Object* const item,
  Date* const data,
  PyArrayObject* const arr)
{
  Date date;
  try {
    date = convert_to_date<Date>(item);
  }
  catch (Exception) {
    return -1;
  }
  std::cerr << "setitem[" << data - (Date const*) PyArray_DATA(arr) << "] = "
            << date << "\n";
  *data = date;
  return 0;
}


//------------------------------------------------------------------------------

template<typename PYDATE>
PyArray_Descr*
DateDtype<PYDATE>::descr_
  = nullptr;


//------------------------------------------------------------------------------

}  // namespace alxs


