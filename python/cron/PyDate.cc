#include <iostream>

#include <Python.h>

#include "py.hh"

#include "PyDate.hh"

namespace aslib {

//------------------------------------------------------------------------------

StructSequenceType*
get_ordinal_date_type()
{
  static StructSequenceType type;

  if (type.tp_name == nullptr) {
    // Lazy one-time initialization.
    static PyStructSequence_Field fields[] = {
      {(char*) "year"       , nullptr},
      {(char*) "ordinal"    , nullptr},
      {nullptr, nullptr}
    };
    static PyStructSequence_Desc desc{
      (char*) "OrdinalDate",                                // name
      nullptr,                                              // doc
      fields,                                               // fields
      2                                                     // n_in_sequence
    };

    StructSequenceType::InitType(&type, &desc);
  }

  return &type;
}


StructSequenceType*
get_week_date_type()
{
  static StructSequenceType type;

  if (type.tp_name == nullptr) {
    // Lazy one-time initialization.
    static PyStructSequence_Field fields[] = {
      {(char*) "week_year"  , nullptr},
      {(char*) "week"       , nullptr},
      {(char*) "weekday"    , nullptr},
      {nullptr, nullptr}
    };
    static PyStructSequence_Desc desc{
      (char*) "WeekDate",                                   // name
      nullptr,                                              // doc
      fields,                                               // fields
      3                                                     // n_in_sequence
    };

    StructSequenceType::InitType(&type, &desc);
  }

  return &type;
}


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
      {nullptr, nullptr}
    };
    static PyStructSequence_Desc desc{
      (char*) "YmdDate",                                    // name
      (char*) docstring::ymddate::type,                     // doc
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
  static ref<Object> months[12];
  static bool initialized = false;
  if (!initialized) {
    // Do a lazy one-time load of the 12 month constants.
    static auto month_type = import("cron", "Month");
    for (cron::Month m = cron::MONTH_MIN; m < cron::MONTH_END; ++m) {
      ref<Tuple> args = Tuple::builder << Long::from(m);
      months[m - 1] = month_type->CallObject(args);
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


ref<Object>
make_ordinal_date(
  cron::OrdinalDate const ordinal_date)
{
  auto ordinal_date_obj = get_ordinal_date_type()->New();
  ordinal_date_obj->initialize(0, Long::from(ordinal_date.year));
  ordinal_date_obj->initialize(1, Long::from(ordinal_date.ordinal));
  return std::move(ordinal_date_obj);
}


ref<Object>
make_week_date(
  cron::WeekDate const week_date)
{
  auto week_date_obj = get_week_date_type()->New();
  week_date_obj->initialize(0, Long::from(week_date.week_year));
  week_date_obj->initialize(1, Long::from(week_date.week));
  week_date_obj->initialize(2, get_weekday_obj(week_date.weekday));
  return std::move(week_date_obj);
}


ref<Object>
make_ymd_date(
  cron::YmdDate const ymd)
{
  auto ymd_obj = get_ymd_date_type()->New();
  ymd_obj->initialize(0, Long::FromLong(ymd.year));
  ymd_obj->initialize(1, get_month_obj(ymd.month));
  ymd_obj->initialize(2, Long::FromLong(ymd.day));
  return std::move(ymd_obj);
}


//------------------------------------------------------------------------------

std::unordered_map<PyTypeObject*, std::unique_ptr<PyDateAPI>>
PyDateAPI::apis_;

//------------------------------------------------------------------------------
// Excplicit template instances

template class PyDate<cron::date::Date>;
template class PyDate<cron::date::Date16>;

//------------------------------------------------------------------------------
// Docstrings

namespace docstring {

namespace pydate {

#include "PyDate.docstrings.cc.inc"

}  // namespace pydate

namespace ymddate {

#include "YmdDate.docstrings.cc.inc"

}  // namespace ymddate

}  // namespace docstring

//------------------------------------------------------------------------------

}  // namespace aslib

