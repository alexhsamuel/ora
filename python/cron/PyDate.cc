#include <iostream>

#include <Python.h>

#include "py.hh"

#include "PyDate.hh"

namespace aslib {

//------------------------------------------------------------------------------

StructSequenceType*
get_ymd_date_type()
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
      (char*) "YmdDate",                                    // name
      nullptr,                                              // doc
      fields,                                               // fields
      3                                                     // n_in_sequence
    };

    StructSequenceType::InitType(&type, &desc);
  }

  return &type;
}


ref<Object>
make_ymd_date(
  cron::YmdDate const ymd)
{
  auto ymd_obj = get_ymd_date_type()->New();
  ymd_obj->initialize(0, Long::FromLong(ymd.year));
  ymd_obj->initialize(1, get_month_obj(ymd.month + 1));
  ymd_obj->initialize(2, Long::FromLong(ymd.day + 1));
  return std::move(ymd_obj);
}


ref<Object>
get_month_obj(
  int month)
{
  static ref<Object> months[12];
  static bool initialized = false;
  if (!initialized) {
    // Do a lazy one-time load of the 12 month constants.
    static auto month_type = import("cron", "Month");
    for (int m = 0; m < 12; ++m) {
      ref<Tuple> args = Tuple::builder << Long::FromLong(m + 1);
      months[m] = month_type->CallObject(args);
    }
    initialized = true;
  }

  return months[month - 1].inc();
}


ref<Object>
get_weekday_obj(
  int weekday)
{
  static ref<Object> weekdays[7];
  static bool initialized = false;
  if (!initialized) {
    // Do a lazy one-time load of the seven weekday constants.
    static auto weekday_type = import("cron", "Weekday");
    for (int w = 0; w < 7; ++w) {
      ref<Tuple> args = Tuple::builder << Long::FromLong(w);
      weekdays[w] = weekday_type->CallObject(args);
    }
    initialized = true;
  }

  return weekdays[weekday].inc();
}


//------------------------------------------------------------------------------

std::unordered_map<PyTypeObject*, std::unique_ptr<PyDateAPI>>
PyDateAPI::apis_;

//------------------------------------------------------------------------------
// Excplicit template instances

template class PyDate<cron::Date>;
template class PyDate<cron::Date16>;

//------------------------------------------------------------------------------

}  // namespace aslib

