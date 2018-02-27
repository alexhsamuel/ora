#include <Python.h>

#include "py.hh"

#include "PyTime.hh"

namespace ora {
namespace py {

//------------------------------------------------------------------------------

StructSequenceType*
get_time_parts_type()
{
  static StructSequenceType type;

  if (type.tp_name == nullptr) {
    // Lazy one-time initialization.
    static PyStructSequence_Field fields[] = {
      {(char*) "date"       , nullptr},
      {(char*) "daytime"    , nullptr},
      {(char*) "time_zone"  , nullptr},
      {nullptr, nullptr}
    };
    static PyStructSequence_Desc desc{
      (char*) "TimeParts",                                  // name
      nullptr,                                              // doc
      fields,                                               // fields
      3                                                     // n_in_sequence
    };

    StructSequenceType::InitType(&type, &desc);
  }

  return &type;
}


//------------------------------------------------------------------------------

std::unordered_map<PyTypeObject*, std::unique_ptr<PyTimeAPI>>
PyTimeAPI::apis_;

//------------------------------------------------------------------------------
// Explicit template instances

template class PyTime<ora::time::Time>;
template class PyTime<ora::time::HiTime>;
template class PyTime<ora::time::SmallTime>;
template class PyTime<ora::time::NsTime>;
template class PyTime<ora::time::Unix32Time>;
template class PyTime<ora::time::Unix64Time>;
template class PyTime<ora::time::Time128>;

//------------------------------------------------------------------------------
// Docstrings

namespace docstring {
namespace pytime {

#include "PyTime.docstrings.cc.inc"

}  // namespace daytime
}  // namespace docstring


//------------------------------------------------------------------------------

}  // namespace py
}  // namespace ora

