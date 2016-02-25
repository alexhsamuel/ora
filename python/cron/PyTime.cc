#include <Python.h>

#include "py.hh"

#include "PyTime.hh"

namespace alxs {

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
      (char*) "TimeParts",                                 // name
      nullptr,                                              // doc
      fields,                                               // fields
      3                                                     // n_in_sequence
    };

    StructSequenceType::InitType(&type, &desc);
  }

  return &type;
}


//------------------------------------------------------------------------------

}  // namespace alxs


