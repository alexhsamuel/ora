#include <Python.h>

#include "py.hh"
#include "util.hh"

namespace ora {
namespace py {

//------------------------------------------------------------------------------

// FIXME: Add time zone info.
StructSequenceType*
get_local_time_type()
{
  static StructSequenceType type;

  if (type.tp_name == nullptr) {
    // Lazy one-time initialization.
    static PyStructSequence_Field fields[] = {
      {(char*) "date"       , nullptr},
      {(char*) "daytime"    , nullptr},
      {nullptr, nullptr}
    };
    static PyStructSequence_Desc desc{
      (char*) "LocalTime",                                  // name
      nullptr,                                              // doc
      fields,                                               // fields
      2                                                     // n_in_sequence
    };

    StructSequenceType::InitType(&type, &desc);
  }

  return &type;
}


//------------------------------------------------------------------------------

}  // namespace py
}  // namespace ora

