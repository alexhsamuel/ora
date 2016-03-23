#pragma once

#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>
#include <numpy/ufuncobject.h>
#include <numpy/npy_3kcompat.h>

#include "mem.hh"
#include "py.hh"

using namespace alxs;

namespace py {
namespace np {

//------------------------------------------------------------------------------

class UFunc
: public Object
{
public:

};


//------------------------------------------------------------------------------

class UnaryUFunc
{
public:

  UnaryUFunc(
    std::string const& name,
    int return_type,
    std::string const& doc="");

  void add_loop(int type_num, PyUFuncGenericFunction fn);

  void add_to_module(Module* const module)
    { module->AddObject(name_.c_str(), ufunc_); }

private:

  /* Ensures that the ufunc object has been created. */
  void ensure();

  std::string const name_;
  int const return_type_;
  std::string const doc_;

  ref<UFunc> ufunc_;

};


inline
UnaryUFunc::UnaryUFunc(
  std::string const& name,
  int const return_type,
  std::string const& doc)
: name_(name),
  return_type_(return_type),
  doc_(doc)
{
}


inline void
UnaryUFunc::add_loop(
  int const type_num,
  PyUFuncGenericFunction const fn)
{
  ensure();

  int arg_types[] = {type_num, return_type_};
  check_zero(
    PyUFunc_RegisterLoopForType(
      (PyUFuncObject*) (UFunc*) ufunc_, 
      type_num,
      fn,
      arg_types,
      nullptr));
}
  

inline void
UnaryUFunc::ensure()
{
  if (ufunc_ == nullptr) {
    ufunc_ = take_not_null<UFunc>(
      // For now, we register the ufunc with no types at all.
      PyUFunc_FromFuncAndData(
        nullptr,              // loop functions
        nullptr,              // data cookie
        nullptr,              // types
        0,                    // number of types
        1,                    // number of arguments
        1,                    // number of return values
        PyUFunc_None,         // identity
        name_.c_str(),        // name
        doc_.c_str(),         // doc
        0));                  // check_return; unused
    // Increment the ref count to make sure the ufunc object isn't destroyed
    // even at shutdown; this happens too late to work correctly.  (?)
    incref(ufunc_);
  }
}


template<typename ARG0, typename RET>
using unary_fn_t = RET (*)(ARG0);


template<typename ARG0, typename RET, unary_fn_t<ARG0, RET> FN>
void
unary_loop_fn(
  char** const args,
  npy_intp* const dimensions,
  npy_intp* const steps,
  void* const /* data */)
{
  auto const n          = dimensions[0];
  auto const ar0_step   = steps[0];
  auto const res_step   = steps[1];
  auto ar0              = (ARG0 const*) args[0];
  auto res              = (RET*) args[1];

  for (npy_intp i = 0; i < n; i++) {
    *res = FN(*ar0);
    ar0 = step(ar0, ar0_step);
    res = step(res, res_step);
  }
}


//------------------------------------------------------------------------------

}  // namespace np
}  // namespace py

