#include <Python.h>

#include "py.hh"

#include "PyDaytime.hh"

namespace ora {
namespace py {

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------

std::unordered_map<PyTypeObject*, std::unique_ptr<PyDaytimeAPI>>
PyDaytimeAPI::apis_;

//------------------------------------------------------------------------------
// Explicit template instances

template class PyDaytime<ora::daytime::Daytime>;
template class PyDaytime<ora::daytime::Daytime32>;
template class PyDaytime<ora::daytime::UsecDaytime>;

//------------------------------------------------------------------------------
// Docstrings

namespace docstring {
namespace pydaytime {

#include "PyDaytime.docstrings.cc.inc"

}  // namespace pydaytime
}  // namespace docstring

//------------------------------------------------------------------------------

}  // namespace py
}  // namespace ora

