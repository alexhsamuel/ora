#include <Python.h>

#include "py.hh"

#include "py_daytime.hh"

namespace ora {
namespace py {

//------------------------------------------------------------------------------

std::unordered_map<PyTypeObject*, std::unique_ptr<PyDaytimeAPI>>
PyDaytimeAPI::apis_;

//------------------------------------------------------------------------------
// Functions

ref<Object>
to_daytime_object(
  Object* obj)
{
  if (PyDaytimeAPI::get(obj) != nullptr)
    return ref<Object>::of(obj);
  else
    return PyDaytimeDefault::create(convert_to_daytime<Daytime>(obj));
}

//------------------------------------------------------------------------------
// Explicit template instances

template class PyDaytime<ora::daytime::Daytime>;
template class PyDaytime<ora::daytime::Daytime32>;
template class PyDaytime<ora::daytime::UsecDaytime>;

//------------------------------------------------------------------------------
// Docstrings

namespace docstring {
namespace pydaytime {

#include "py_daytime.docstrings.cc.inc"

}  // namespace pydaytime
}  // namespace docstring

//------------------------------------------------------------------------------

}  // namespace py
}  // namespace ora

