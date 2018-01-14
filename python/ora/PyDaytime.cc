#include <Python.h>

#include "py.hh"

#include "PyDaytime.hh"

namespace ora {
namespace py {

//------------------------------------------------------------------------------

StructSequenceType*
get_hms_daytime_type()
{
  static StructSequenceType type;

  if (type.tp_name == nullptr) {
    // Lazy one-time initialization.
    static PyStructSequence_Field fields[] = {
      {(char*) "hour"       , nullptr},
      {(char*) "minute"     , nullptr},
      {(char*) "second"     , nullptr},
      {nullptr, nullptr}
    };
    static PyStructSequence_Desc desc{
      (char*) "HmsDaytime",                                 // name
      nullptr,                                              // doc
      fields,                                               // fields
      3                                                     // n_in_sequence
    };

    StructSequenceType::InitType(&type, &desc);
  }

  return &type;
}


ref<Object>
make_hms_daytime(
  ora::HmsDaytime const hms)
{
  auto hms_obj = get_hms_daytime_type()->New();
  hms_obj->initialize(0, Long::FromLong(hms.hour));
  hms_obj->initialize(1, Long::FromLong(hms.minute));
  hms_obj->initialize(2, Float::FromDouble(hms.second));
  return std::move(hms_obj);
}


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

