#pragma once

#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>
#include <numpy/ufuncobject.h>
#include <numpy/npy_3kcompat.h>

#include "aslib/mem.hh"
#include "py.hh"

using namespace aslib;

namespace py {
namespace np {

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
  return std::move(ufunc);
}


//------------------------------------------------------------------------------

/*
 * Wraps a unary function `FN(ARG0) -> RET0` in a ufunc loop function
 */
template<typename ARG0, typename RET0, RET0 (*FN)(ARG0)>
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


//------------------------------------------------------------------------------

}  // namespace np
}  // namespace py

