#include <Python.h>

#include "numpy.hh"
#include "ora/lib/mem.hh"
#include "ora.hh"
#include "py.hh"

namespace ora {
namespace py {

using namespace np;

//------------------------------------------------------------------------------

// FIXME: Organize headers better.
template<class TIME> inline std::pair<bool, TIME> maybe_time(Object*);
template<class TIME> inline TIME convert_to_time(Object*);

//------------------------------------------------------------------------------

template<class PYTIME>
class TimeDtype
{
public:

  using Time = typename PYTIME::Time;

  static void set_up_dtype();

private:

  static Object*        getitem(Time const*, PyArrayObject*);
  static int            setitem(Object*, Time*, PyArrayObject*);
  static int            compare(Time const*, Time const*, PyArrayObject*);

  static void           cast_from_object(Object* const*, Time*, npy_intp, void*, void*);

  static npy_bool equal(Time const time0, Time const time1) 
    { return ora::time::nex::equal(time0, time1) ? NPY_TRUE : NPY_FALSE; }
  static npy_bool not_equal(Time const time0, Time const time1)
    { return ora::time::nex::equal(time0, time1) ? NPY_FALSE : NPY_TRUE; }

  static PyArray_Descr* descr_;

};


template<class PYTIME>
void
TimeDtype<PYTIME>::set_up_dtype()
{
  assert(descr_ == nullptr);

  // Deliberately 'leak' this instance, as it has process lifetime.
  auto arr_funcs = new PyArray_ArrFuncs;
  PyArray_InitArrFuncs(arr_funcs);
  arr_funcs->copyswap   = (PyArray_CopySwapFunc*) generic_copyswap<Time>;
  arr_funcs->copyswapn  = (PyArray_CopySwapNFunc*) generic_copyswapn<Time>;
  arr_funcs->getitem    = (PyArray_GetItemFunc*) getitem;
  arr_funcs->setitem    = (PyArray_SetItemFunc*) setitem;
  arr_funcs->compare    = (PyArray_CompareFunc*) compare;
  // FIMXE: Additional methods.

  descr_ = PyObject_New(PyArray_Descr, &PyArrayDescr_Type);
  descr_->typeobj       = incref(&PYTIME::type_);
  descr_->kind          = 'V';
  descr_->type          = 'j';  // FIXME?
  descr_->byteorder     = '=';
  // FIXME: Requires initialization to INVALID.  Or else Time needs to handle
  // any bit pattern correctly.
  descr_->flags         = 0;
  descr_->type_num      = 0;
  descr_->elsize        = sizeof(Time);
  descr_->alignment     = alignof(Time);
  descr_->subarray      = nullptr;
  descr_->fields        = nullptr;
  descr_->names         = nullptr;
  descr_->f             = arr_funcs;
  descr_->metadata      = nullptr;
  descr_->c_metadata    = nullptr;
  descr_->hash          = -1;

  if (PyArray_RegisterDataType(descr_) < 0)
    throw py::Exception();

  // Set the dtype as an attribute to the scalar type.
  assert(PYTIME::type_.tp_dict != nullptr);
  ((Dict*) PYTIME::type_.tp_dict)->SetItemString("dtype", (Object*) descr_);

  auto const npy_object = PyArray_DescrFromType(NPY_OBJECT);
  auto const np_module = Module::ImportModule("numpy");

  if (PyArray_RegisterCastFunc(
        npy_object, descr_->type_num, 
        (PyArray_VectorUnaryFunc*) cast_from_object) < 0)
    throw py::Exception();
  if (PyArray_RegisterCanCast(
        npy_object, descr_->type_num, NPY_OBJECT_SCALAR) < 0)
    throw py::Exception();

  create_or_get_ufunc(np_module, "equal", 2, 1)->add_loop_2(
    descr_->type_num, descr_->type_num, NPY_BOOL,
    ufunc_loop_2<Time, Time, npy_bool, equal>);
  create_or_get_ufunc(np_module, "not_equal", 2, 1)->add_loop_2(
    descr_->type_num, descr_->type_num, NPY_BOOL,
    ufunc_loop_2<Time, Time, npy_bool, not_equal>);
}


template<class PYTIME>
Object*
TimeDtype<PYTIME>::getitem(
  Time const* const data,
  PyArrayObject* const arr)
{
  return PYTIME::create(*data).release();
}


template<class PYTIME>
int
TimeDtype<PYTIME>::setitem(
  Object* const item,
  Time* const data,
  PyArrayObject* const arr)
{
  try {
    *data = convert_to_time<Time>(item);
  }
  catch (Exception) {
    return -1;
  }
  return 0;
}


template<class PYTIME>
int 
TimeDtype<PYTIME>::compare(
  Time const* const t0, 
  Time const* const t1, 
  PyArrayObject* const /* arr */)
{
  return 
      t0->is_invalid() ? -1
    : t1->is_invalid() ?  1
    : t0->is_missing() ? -1
    : t1->is_missing() ?  1
    : *t0 < *t1        ? -1 
    : *t0 > *t1        ?  1 
    : 0;
}


template<class PYTIME>
void
TimeDtype<PYTIME>::cast_from_object(
  Object* const* from,
  Time* to,
  npy_intp num,
  void* /* unused */,
  void* /* unused */)
{
  for (; num > 0; --num, ++from, ++to) {
    auto const time = maybe_time<Time>(*from);
    *to = time.first ? time.second : Time::INVALID;
  }
}


//------------------------------------------------------------------------------

template<class PYTIME>
PyArray_Descr*
TimeDtype<PYTIME>::descr_
  = nullptr;

//------------------------------------------------------------------------------

}  // namespace py
}  // namespace ora
