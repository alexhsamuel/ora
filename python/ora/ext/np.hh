#pragma once

#define PY_ARRAY_UNIQUE_SYMBOL ora_PyArray_API
#define PY_UFUNC_UNIQUE_SYMBOL ora_PyUFunc_API
#define NO_IMPORT
#define NO_IMPORT_ARRAY
#define NO_IMPORT_UFUNC
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>
#include <numpy/ufuncobject.h>
#include <numpy/npy_3kcompat.h>

#include "ora/lib/mem.hh"
#include "py.hh"

namespace ora {
namespace py {
namespace np {

using namespace ora::lib;

//------------------------------------------------------------------------------

inline void
check_succeed(
  npy_intp status)
{
  if (status != NPY_SUCCEED)
    throw Exception();
}


//------------------------------------------------------------------------------

class Descr
: public PyArray_Descr
{
public:

  static Descr* from(int const typenum)
    { return (Descr*) PyArray_DescrFromType(typenum); }
  
};


class Array
: public Object
{
public:

  static bool Check(PyObject* const obj)
    { return PyArray_Check(obj); }
  static ref<Array> FromAny(PyObject* const obj, PyArray_Descr* const dtype, int const dims_min, int const dims_max, int const requirements, PyObject* const context=nullptr)
    { return take_not_null<Array>(PyArray_FromAny(obj, dtype, dims_min, dims_max, requirements, context)); }
  static ref<Array> FromAny(PyObject* const obj, int const dtype, int const dims_min, int const dims_max, int const requirements, PyObject* const context=nullptr)
    { return FromAny(obj, PyArray_DescrFromType(dtype), dims_min, dims_max, requirements, context); }
  static void RegisterCanCast(PyArray_Descr* const from, int const to, NPY_SCALARKIND const scalar)
    { check_zero(PyArray_RegisterCanCast(from, to, scalar)); }
  static void RegisterCanCast(int const from, int const to, NPY_SCALARKIND const scalar)
    { check_zero(PyArray_RegisterCanCast(PyArray_DescrFromType(from), to, scalar)); }
  static void RegisterCastFunc(PyArray_Descr* const from, int const to, PyArray_VectorUnaryFunc* const f)
    { check_zero(PyArray_RegisterCastFunc(from, to, f)); }
  static void RegisterCastFunc(int const from, int const to, PyArray_VectorUnaryFunc* const f)
    { check_zero(PyArray_RegisterCastFunc(PyArray_DescrFromType(from), to, f)); }
  static ref<Array> NewLikeArray(Array* prototype, NPY_ORDER order=NPY_CORDER, PyArray_Descr* descr=nullptr, bool subok=true)
    { return take_not_null<Array>(PyArray_NewLikeArray((PyArrayObject*) prototype, order, (PyArray_Descr*) xincref((PyObject*) descr), subok ? 1 : 0)); }
  static ref<Array> SimpleNew(int const nd, npy_intp* const dims, int const typenum)
    { return take_not_null<Array>(PyArray_SimpleNew(nd, dims, typenum)); }
  static ref<Array> SimpleNew1D(npy_intp const size, int const typenum)
    { return SimpleNew(1, const_cast<npy_intp*>(&size), typenum); }

  PyArray_Descr* descr()
    { return PyArray_DESCR(array_this()); }
  npy_intp size()
    { return PyArray_SIZE(array_this()); }
  template<class T> T const* get_const_ptr()
    { return reinterpret_cast<T*>(PyArray_DATA(array_this())); }
  template<class T> T* get_ptr()
    { return reinterpret_cast<T*>(PyArray_DATA(array_this())); }

private:

  // The way we have things set up right now, we can't have Array derive both
  // Object and PyArrayObject.
  // FIXME: Hackamoley.
  PyArrayObject* array_this() { return reinterpret_cast<PyArrayObject*>(this); }

};


template<class FROM, class TO, TO (*FUNC)(FROM)>
void
cast_func(
  FROM const* from,
  TO* to,
  npy_intp num,
  void* /* unused */,
  void* /* unused */)
{
  for (; num > 0; --num, ++from, ++to)
    *to = FUNC(*from);
}


//------------------------------------------------------------------------------

// Compile-time mapping from C++ integer types to numpy type numbers.
template<class INT> struct IntType 
  { static int constexpr type_num = -1; };
template<> struct IntType<int16_t> 
  { static int constexpr type_num = NPY_INT16; };
template<> struct IntType<uint16_t>
  { static int constexpr type_num = NPY_UINT16; };
template<> struct IntType<int32_t> 
  { static int constexpr type_num = NPY_INT32; };
template<> struct IntType<uint32_t>
  { static int constexpr type_num = NPY_UINT32; };
template<> struct IntType<int64_t>
  { static int constexpr type_num = NPY_INT64; };
template<> struct IntType<uint64_t>
  { static int constexpr type_num = NPY_UINT64; };

//------------------------------------------------------------------------------

template<class TYPE>
class ArrayIter
: public Object
{
public:

  TYPE& operator*()
    { return *(TYPE*) (obj()->dataptr); }
  TYPE operator*() const
    { return *(TYPE const*) (obj()->dataptr); }

private:

  ArrayIter() = delete;
  ArrayIter(ArrayIter const&) = delete;
  ArrayIter(ArrayIter&&) = delete;

  PyArrayIterObject* obj() const
    { return (PyArrayIterObject*) this; }

};


class ArrayMultiIter
: public Object
{
public:

  static ref<ArrayMultiIter> New(PyObject* obj0, PyObject* obj1, PyObject* obj2)
    { return take_not_null<ArrayMultiIter>(PyArray_MultiIterNew(3, obj0, obj1, obj2)); }

  operator bool() const
    { return PyArray_MultiIter_NOTDONE(this); }
  bool not_done() const
    { return PyArray_MultiIter_NOTDONE(this); }
  void next()
    { PyArray_MultiIter_NEXT(this); }

  int nd() const
    { return obj()->nd; }
  npy_intp* dimensions() const
    { return obj()->dimensions; }
  template<class TYPE> ArrayIter<TYPE>& iter(size_t const index)
    { return *(ArrayIter<TYPE>*) (obj()->iters[index]); }
  npy_intp index() const
    { return obj()->index; }

private:

  ArrayMultiIter() = delete;
  ArrayMultiIter(ArrayMultiIter const&) = delete;
  ArrayMultiIter(ArrayMultiIter&&) = delete;

  PyArrayMultiIterObject* obj() const
    { return (PyArrayMultiIterObject*) this; }

};

//------------------------------------------------------------------------------

class UFunc
: public Object
{
public:

  /*
   * Creates and returns a ufunc with no loop functions.
   */
  static ref<UFunc> create_empty(
    char const* name, unsigned char num_args, unsigned char num_ret,
    char const* doc=nullptr, int identity=PyUFunc_None);

  /*
   * Adds a loop function with one argument and one return.
   */
  void add_loop_1(int arg0_type, int ret0_type, PyUFuncGenericFunction);
  void add_loop_1(PyArray_Descr*, PyArray_Descr*, PyUFuncGenericFunction);

  void add_loop_2(int arg0_type, int arg1_type, int ret0_type, PyUFuncGenericFunction);
  void add_loop_2(PyArray_Descr*, PyArray_Descr*, PyArray_Descr*, PyUFuncGenericFunction);

};


inline ref<UFunc>
UFunc::create_empty(
  char const*   const name,
  unsigned char const num_args,
  unsigned char const num_rets,
  char const*   const doc,
  int           const identity)
{
  return take_not_null<UFunc>(
    PyUFunc_FromFuncAndData(
      nullptr,              // loop functions
      nullptr,              // data cookie
      nullptr,              // types
      0,                    // number of types
      num_args,             // number of arguments
      num_rets,             // number of return values
      identity,             // identity
      (char*) name,         // name
      (char*) doc,          // doc
      0));                  // check_return; unused
}


inline void
UFunc::add_loop_1(
  int const arg0_type,
  int const ret0_type,
  PyUFuncGenericFunction const fn)
{
  // FIXME: Check that num_args == 1 and num_rets == 1.

  int arg_types[] = {arg0_type, ret0_type};
  check_zero(
    PyUFunc_RegisterLoopForType(
      (PyUFuncObject*) this,
      arg0_type,
      fn,
      arg_types,
      nullptr));
}


inline void
UFunc::add_loop_1(
  PyArray_Descr* const arg0_dtype,
  PyArray_Descr* const ret0_dtype,
  PyUFuncGenericFunction const fn)
{
  // FIXME: Check that num_args == 1 and num_rets == 1.

  PyArray_Descr* dtypes[] = {arg0_dtype, ret0_dtype};
  check_zero(
    PyUFunc_RegisterLoopForDescr(
      (PyUFuncObject*) this,
      arg0_dtype,
      fn,
      dtypes,
      nullptr));
}


inline void
UFunc::add_loop_2(
  int const arg0_type,
  int const arg1_type,
  int const ret0_type,
  PyUFuncGenericFunction const fn)
{
  // FIXME: Check that num_args == 2 and num_rets == 1.

  int arg_types[] = {arg0_type, arg1_type, ret0_type};
  check_zero(
    PyUFunc_RegisterLoopForType(
      (PyUFuncObject*) this,
      arg0_type >= NPY_USERDEF ? arg0_type : arg1_type,
      fn,
      arg_types,
      nullptr));
}


inline void
UFunc::add_loop_2(
  PyArray_Descr* const arg0_dtype,
  PyArray_Descr* const arg1_dtype,
  PyArray_Descr* const ret0_dtype,
  PyUFuncGenericFunction const fn)
{
  // FIXME: Check that num_args == 2 and num_rets == 1.

  PyArray_Descr* dtypes[] = {arg0_dtype, arg1_dtype, ret0_dtype};
  check_zero(
    PyUFunc_RegisterLoopForDescr(
      (PyUFuncObject*) this,
      arg0_dtype,
      fn,
      dtypes,
      nullptr));
}


/*
 * Gets a ufunc from a module; if not found, creates it.
 *
 * The ufunc's name in the module matches its internal name.
 */
inline ref<UFunc>
create_or_get_ufunc(
  Module*       const module,
  char const*   const name,
  unsigned char const num_args,
  unsigned char const num_rets,
  char const*   const doc=nullptr)
{
  ref<UFunc> ufunc = cast<UFunc>(module->GetAttrString(name, false));
  if (ufunc == nullptr) {
    ufunc = UFunc::create_empty(name, num_args, num_rets, doc);
    module->AddObject(name, ufunc);
  }
  else
    // FIXME: Check name, num_args, num_rets.
    ;
  return ufunc;
}


//------------------------------------------------------------------------------

/*
 * Wraps a unary function `FN(ARG0) -> RET0` in a ufunc loop function
 */
template<class ARG0, class RET0, RET0 (*FN)(ARG0)>
void
ufunc_loop_1(
  char** const args,
  npy_intp* const dimensions,
  npy_intp* const steps,
  void* const /* data */)
{
  auto const n          = dimensions[0];
  auto const arg0_step  = steps[0];
  auto const ret0_step  = steps[1];
  auto arg0             = (ARG0 const*) args[0];
  auto ret0             = (RET0*) args[1];

  for (npy_intp i = 0; i < n; i++) {
    *ret0 = FN(*arg0);
    arg0 = step(arg0, arg0_step);
    ret0 = step(ret0, ret0_step);
  }
}


/*
 * Wraps a unary function `FN(ARG0, ARG1) -> RET0` in a ufunc loop function
 */
template<class ARG0, class ARG1, class RET0, RET0 (*FN)(ARG0, ARG1)>
void
ufunc_loop_2(
  char** const args,
  npy_intp* const dimensions,
  npy_intp* const steps,
  void* const /* data */)
{
  auto const n          = dimensions[0];
  auto const arg0_step  = steps[0];
  auto const arg1_step  = steps[1];
  auto const ret0_step  = steps[2];
  auto arg0             = (ARG0 const*) args[0];
  auto arg1             = (ARG1 const*) args[1];
  auto ret0             = (RET0*) args[2];

  for (npy_intp i = 0; i < n; i++) {
    *ret0 = FN(*arg0, *arg1);
    arg0 = step(arg0, arg0_step);
    arg1 = step(arg1, arg1_step);
    ret0 = step(ret0, ret0_step);
  }
}


//------------------------------------------------------------------------------

/*
 * Registers comparison ufunc loops.
 */
template<class TYPE, bool (*EQUAL)(TYPE, TYPE), bool (*LESS)(TYPE, TYPE)>
class Comparisons
{
public:

  static void
  register_loops(
    npy_intp const type_num)
  {
    auto const np_module = Module::ImportModule("numpy");

    create_or_get_ufunc(np_module, "equal", 2, 1)->add_loop_2(
      type_num, type_num, NPY_BOOL,
      ufunc_loop_2<TYPE, TYPE, npy_bool, equal>);
    create_or_get_ufunc(np_module, "not_equal", 2, 1)->add_loop_2(
      type_num, type_num, NPY_BOOL,
      ufunc_loop_2<TYPE, TYPE, npy_bool, not_equal>);
    create_or_get_ufunc(np_module, "less", 2, 1)->add_loop_2(
      type_num, type_num, NPY_BOOL,
      ufunc_loop_2<TYPE, TYPE, npy_bool, less>);
    create_or_get_ufunc(np_module, "less_equal", 2, 1)->add_loop_2(
      type_num, type_num, NPY_BOOL,
      ufunc_loop_2<TYPE, TYPE, npy_bool, less_equal>);
    create_or_get_ufunc(np_module, "greater", 2, 1)->add_loop_2(
      type_num, type_num, NPY_BOOL,
      ufunc_loop_2<TYPE, TYPE, npy_bool, greater>);
    create_or_get_ufunc(np_module, "greater_equal", 2, 1)->add_loop_2(
      type_num, type_num, NPY_BOOL,
      ufunc_loop_2<TYPE, TYPE, npy_bool, greater_equal>);
  }

private:

  Comparisons() = delete;

  static npy_bool equal(TYPE const a, TYPE const b)
    { return EQUAL(a, b) ? NPY_TRUE : NPY_FALSE; }

  static npy_bool not_equal(TYPE const a, TYPE const b)
    { return EQUAL(a, b) ? NPY_FALSE : NPY_TRUE; }

  static npy_bool less(TYPE const a, TYPE const b)
    { return LESS(a, b) ? NPY_TRUE : NPY_FALSE; }

  static npy_bool less_equal(TYPE const a, TYPE const b)
    { return EQUAL(a, b) || LESS(a, b) ? NPY_TRUE : NPY_FALSE; }

  static npy_bool greater(TYPE const a, TYPE const b)
    { return EQUAL(a, b) || LESS(a, b) ? NPY_FALSE : NPY_TRUE; }

  static npy_bool greater_equal(TYPE const a, TYPE const b)
    { return LESS(a, b) ? NPY_FALSE : NPY_TRUE; }

};


//------------------------------------------------------------------------------

template<class TYPE>
void
generic_copyswap(
  TYPE* const dst,
  TYPE const* const src,
  int const swap,
  PyArrayObject* const arr)
{
  if (swap)
    copy_swapped<sizeof(TYPE)>(src, dst);
  else
    copy<sizeof(TYPE)>(src, dst);
}


template<class TYPE>
void
generic_copyswapn(
  TYPE* const dst, 
  npy_intp const dst_stride, 
  TYPE const* const src, 
  npy_intp const src_stride, 
  npy_intp const n, 
  int const swap, 
  PyArrayObject* const arr)
{
  if (src_stride == 0) {
    // Special case: swapped or unswapped fill.
    TYPE val;
    if (swap) 
      copy_swapped<sizeof(TYPE)>(src, &val);
    else
      val = *src;

    char* d = (char*) dst;
    for (npy_intp i = 0; i < n; ++i) {
      *(TYPE*) d = val;
      d += dst_stride;
    }
  }

  else {
    char const* s = (char const*) src;
    char* d = (char*) dst;
    if (swap) 
      for (npy_intp i = 0; i < n; ++i) {
        copy_swapped<sizeof(TYPE)>(s, d);
        s += src_stride;
        d += dst_stride;
      }
    else 
      for (npy_intp i = 0; i < n; ++i) {
        copy<sizeof(TYPE)>(s, d);
        s += src_stride;
        d += dst_stride;
      }
  }
}


//------------------------------------------------------------------------------

}  // namespace np
}  // namespace py
}  // namespace ora

