#pragma once

#define PY_ARRAY_UNIQUE_SYMBOL ora_numpy
#define NO_IMPORT_ARRAY
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
  static ref<Array> SimpleNew(int const nd, npy_intp* const dims, int const typenum)
    { return take_not_null<Array>(PyArray_SimpleNew(nd, dims, typenum)); }
  static ref<Array> SimpleNew1D(npy_intp const size, int const typenum)
    { return SimpleNew(1, const_cast<npy_intp*>(&size), typenum); }

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
      arg0_type,
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

}  // namespace np
}  // namespace py
}  // namespace ora

