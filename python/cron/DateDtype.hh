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
  static Object*        getitem(void*, void*);
  static int            setitem(Object*, void*, void*);
  static void           copyswap(void*, void*, int, void*);

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
    arr_funcs->getitem      = (PyArray_GetItemFunc*) getitem;
    arr_funcs->setitem      = (PyArray_SetItemFunc*) setitem;
    arr_funcs->copyswap     = (PyArray_CopySwapFunc*) copyswap;

    descr_ = PyObject_New(PyArray_Descr, &PyArrayDescr_Type);
    descr_->typeobj         = incref(&PYDATE::type_);
    descr_->kind            = 'V';
    descr_->type            = 'j';
    descr_->byteorder       = '=';
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
  auto const dict = (Dict*) dtype->typeobj->tp_dict;
  assert(dict != nullptr);
  dict->SetItemString("dtype", (Object*) dtype);
}


template<typename PYDATE>
Object*
DateDtype<PYDATE>::getitem(
  void* const data,
  void* const /* arr */)
{
  // FIXME: Check PyArray_ISBEHAVED_RO(arr)?
  return PYDATE::create(*reinterpret_cast<Date const*>(data)).release();
}


template<typename PYDATE>
int
DateDtype<PYDATE>::setitem(
  Object* const item,
  void* const data,
  void* const /* arr */)
{
  Date date;
  try {
    date = convert_to_date<Date>(item);
  }
  catch (Exception) {
    return -1;
  }
  *reinterpret_cast<Date*>(data) = date;
  return 0;
}


template<typename PYDATE>
void
DateDtype<PYDATE>::copyswap(
  void* const dest,
  void* const src,
  int const swap,
  void* const /* arr */)
{
  assert(!swap);  // FIXME
  *reinterpret_cast<Date*>(dest) = *reinterpret_cast<Date const*>(src);
}


template<typename PYDATE>
PyArray_Descr*
DateDtype<PYDATE>::descr_
  = nullptr;


//------------------------------------------------------------------------------

}  // namespace alxs


