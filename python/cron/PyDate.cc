#include <iostream>

#include <Python.h>

#include "py.hh"

#include "PyDate.hh"

namespace alxs {

//------------------------------------------------------------------------------

StructSequenceType*
get_date_parts_type()
{
  static StructSequenceType type;

  if (type.tp_name == nullptr) {
    // Lazy one-time initialization.
    static PyStructSequence_Field fields[] = {
      {(char*) "year"       , nullptr},
      {(char*) "month"      , nullptr},
      {(char*) "day"        , nullptr},
      {(char*) "ordinal"    , nullptr},
      {(char*) "week_year"  , nullptr},
      {(char*) "week"       , nullptr},
      {(char*) "weekday"    , nullptr},
      {nullptr, nullptr}
    };
    static PyStructSequence_Desc desc{
      (char*) "DateParts",                                  // name
      nullptr,                                              // doc
      fields,                                               // fields
      7                                                     // n_in_sequence
    };

    StructSequenceType::InitType(&type, &desc);
  }

  return &type;
}


//------------------------------------------------------------------------------

}  // namespace alxs

