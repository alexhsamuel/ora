#include <Python.h>

#include "py.hh"

#include "PyDaytime.hh"

namespace aslib {

//------------------------------------------------------------------------------

StructSequenceType*
get_daytime_parts_type()
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
      (char*) "DaytimeParts",                               // name
      nullptr,                                              // doc
      fields,                                               // fields
      3                                                     // n_in_sequence
    };

    StructSequenceType::InitType(&type, &desc);
  }

  return &type;
}


//------------------------------------------------------------------------------

}  // namespace aslib

