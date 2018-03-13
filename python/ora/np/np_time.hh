#pragma once

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

/*
 * Dispatch for non-ufunc functions to time type-specific implementation.
 */
class TimeAPI
{
public:

  virtual ~TimeAPI() {}
  virtual ref<Object> from_offset(Array*) = 0;

  static TimeAPI* get(PyArray_Descr* descr) {
    assert(descr->c_metadata != nullptr);
    return reinterpret_cast<TimeAPI*>(descr->c_metadata); 
  }
    
};


// FIXME: We should just subclass PyArray_Descr!
template<class PYTIME>
class TimeDtype
{
public:

  using Time = typename PYTIME::Time;
  using Offset = typename Time::Offset;

  static void set_up_dtype(Module*);
  static Descr* get_descr() { return descr_; }  // FIXME: Subclass!

private:

  static Object*        getitem(Time const*, PyArrayObject*);
  static int            setitem(Object*, Time*, PyArrayObject*);
  static int            compare(Time const*, Time const*, PyArrayObject*);

  static void           cast_from_object(Object* const*, Time*, npy_intp, void*, void*);

  static npy_bool equal(Time const time0, Time const time1) 
    { return ora::time::nex::equal(time0, time1) ? NPY_TRUE : NPY_FALSE; }
  static npy_bool not_equal(Time const time0, Time const time1)
    { return ora::time::nex::equal(time0, time1) ? NPY_FALSE : NPY_TRUE; }

  class API
  : public TimeAPI
  {
  public:

    virtual ~API() = default;
    virtual ref<Object> from_offset(Array*) override;

  };

  static Descr* descr_;

};


template<class PYTIME>
void
TimeDtype<PYTIME>::set_up_dtype(
  Module* module)
{
  assert(descr_ == nullptr);
  assert(module != nullptr);

  // Deliberately 'leak' this instance, as it has process lifetime.
  auto arr_funcs = new PyArray_ArrFuncs;
  PyArray_InitArrFuncs(arr_funcs);
  arr_funcs->copyswap   = (PyArray_CopySwapFunc*) generic_copyswap<Time>;
  arr_funcs->copyswapn  = (PyArray_CopySwapNFunc*) generic_copyswapn<Time>;
  arr_funcs->getitem    = (PyArray_GetItemFunc*) getitem;
  arr_funcs->setitem    = (PyArray_SetItemFunc*) setitem;
  arr_funcs->compare    = (PyArray_CompareFunc*) compare;
  // FIMXE: Additional methods.

  descr_ = (Descr*) PyObject_New(PyArray_Descr, &PyArrayDescr_Type);
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
  descr_->c_metadata    = (NpyAuxData*) new API();
  descr_->hash          = -1;

  if (PyArray_RegisterDataType(descr_) < 0)
    throw py::Exception();
  int const type_num = descr_->type_num;

  // Set the dtype as an attribute to the scalar type.
  assert(PYTIME::type_.tp_dict != nullptr);
  ((Dict*) PYTIME::type_.tp_dict)->SetItemString("dtype", (Object*) descr_);

  auto const np_module = Module::ImportModule("numpy");

  int constexpr int_type_num = IntType<Offset>::type_num;

  // Cast from object to time.
  Array::RegisterCastFunc(
    NPY_OBJECT, type_num, (PyArray_VectorUnaryFunc*) cast_from_object);
  Array::RegisterCanCast(NPY_OBJECT, type_num, NPY_OBJECT_SCALAR);

  create_or_get_ufunc(np_module, "equal", 2, 1)->add_loop_2(
    type_num, type_num, NPY_BOOL,
    ufunc_loop_2<Time, Time, npy_bool, equal>);
  create_or_get_ufunc(np_module, "not_equal", 2, 1)->add_loop_2(
    type_num, type_num, NPY_BOOL,
    ufunc_loop_2<Time, Time, npy_bool, not_equal>);

  if (int_type_num != -1) {
    create_or_get_ufunc(module, "to_offset", 1, 1)->add_loop_1(
      type_num, int_type_num,
      ufunc_loop_1<Time, Offset, ora::time::nex::get_offset<Time>>);
  }

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
// API implementation

template<class PYTIME>
ref<Object>
TimeDtype<PYTIME>::API::from_offset(
  Array* const offset)
{
  size_t constexpr nargs = 2;
  PyArrayObject* op[nargs] = {(PyArrayObject*) offset, nullptr};
  // Tell the iterator to allocate the output automatically.
  npy_uint32 flags[nargs] 
    = {NPY_ITER_READONLY, NPY_ITER_WRITEONLY | NPY_ITER_ALLOCATE};
  PyArray_Descr* dtypes[nargs] = {Descr::from(NPY_INT64), descr_};

  // Construct the iterator.  We'll handle the inner loop explicitly.
  auto const iter = NpyIter_MultiNew(
    nargs, op, NPY_ITER_EXTERNAL_LOOP, NPY_KEEPORDER, NPY_UNSAFE_CASTING, 
    flags, dtypes);
  if (iter == nullptr)
    throw Exception();

  auto const next = NpyIter_GetIterNext(iter, nullptr);
  auto const inner_stride = NpyIter_GetInnerStrideArray(iter)[0];
  auto const item_size = NpyIter_GetDescrArray(iter)[1]->elsize;

  auto const& inner_size = *NpyIter_GetInnerLoopSizePtr(iter);
  auto const data_ptrs = NpyIter_GetDataPtrArray(iter);

  do {
    // Note: Since dst is newly allocated, it is tightly packed.
    auto src = data_ptrs[0];
    auto dst = data_ptrs[1];
    for (auto size = inner_size; 
         size > 0; 
         --size, src += inner_stride, dst += item_size)
      *reinterpret_cast<Time*>(dst) 
        = ora::time::nex::from_offset<Time>(*reinterpret_cast<int64_t*>(src));
  } while (next(iter));

  // Get the result from the iterator object array.
  auto ret = ref<Array>::of((Array*) NpyIter_GetOperandArray(iter)[1]);
  check_succeed(NpyIter_Deallocate(iter));
  return std::move(ret);
}


//------------------------------------------------------------------------------

template<class PYTIME>
Descr*
TimeDtype<PYTIME>::descr_
  = nullptr;

//------------------------------------------------------------------------------

}  // namespace py
}  // namespace ora
