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
      3                                                     // n_in_sequence
    };

    StructSequenceType::InitType(&type, &desc);
  }

  return &type;
}


ref<Object>
get_month_obj(
  int month)
{
  static auto month_type = import("cron", "Month");
  ref<Tuple> args = Tuple::builder << Long::FromLong(month);
  return month_type->CallObject(args);
}


ref<Object>
get_weekday_obj(
  int weekday)
{
  static auto weekday_type = import("cron", "Weekday");
  ref<Tuple> args = Tuple::builder << Long::FromLong(weekday);
  return weekday_type->CallObject(args);
}


//------------------------------------------------------------------------------

std::unordered_map<PyTypeObject*, std::unique_ptr<PyDateAPI>>
PyDateAPI::apis_;

//------------------------------------------------------------------------------

}  // namespace alxs

