#pragma once

#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>
#include <numpy/ufuncobject.h>
#include <numpy/npy_3kcompat.h>

#include "py.hh"

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


//------------------------------------------------------------------------------

}  // namespace np
}  // namespace py

