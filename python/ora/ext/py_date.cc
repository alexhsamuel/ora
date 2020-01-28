#include <iostream>

#include <Python.h>

#include "ora/format.hh"
#include "py_date.hh"
#include "py.hh"

namespace ora {
namespace py {

//------------------------------------------------------------------------------
// Functions

ref<Object>
to_date_object(
  Object* obj)
{
  if (PyDateAPI::get(obj) != nullptr)
    return ref<Object>::of(obj);
  else
    return PyDateDefault::create(convert_to_date<Date>(obj));
}


Weekday
convert_to_weekday(
  Object* obj)
{
  static auto weekday_type = import("ora", "Weekday");
  ref<Tuple> args = Tuple::builder << ref<Object>::of(obj);
  // FIXME
  auto weekday = ref<Object>::take(PyObject_CallObject(weekday_type, args));
  if (weekday != nullptr)
     return weekday->long_value();

  auto str = weekday->Str()->as_utf8_string();
  return parse_weekday_name(str);
}


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


ref<Object>
get_month_obj(
  int month)
{
  static ref<Object> months[12];
  static bool initialized = false;
  if (!initialized) {
    // Do a lazy one-time load of the 12 month constants.
    static auto month_type = import("ora", "Month");
    for (ora::Month m = ora::MONTH_MIN; m < ora::MONTH_END; ++m) {
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
    static auto weekday_type = import("ora", "Weekday");
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
  ora::OrdinalDate const ordinal_date)
{
  auto ordinal_date_obj = get_ordinal_date_type()->New();
  ordinal_date_obj->initialize(0, Long::from(ordinal_date.year));
  ordinal_date_obj->initialize(1, Long::from(ordinal_date.ordinal));
  return std::move(ordinal_date_obj);
}


ref<Object>
make_week_date(
  ora::WeekDate const week_date)
{
  auto week_date_obj = get_week_date_type()->New();
  week_date_obj->initialize(0, Long::from(week_date.week_year));
  week_date_obj->initialize(1, Long::from(week_date.week));
  week_date_obj->initialize(2, get_weekday_obj(week_date.weekday));
  return std::move(week_date_obj);
}


ref<Object>
make_ymd_date(
  ora::YmdDate const ymd)
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

template class PyDate<ora::date::Date>;
template class PyDate<ora::date::Date16>;

//------------------------------------------------------------------------------
// Docstrings

namespace docstring {
namespace pydate {

#include "py_date.docstrings.cc.inc"

}  // namespace pydate
}  // namespace docstring

//------------------------------------------------------------------------------

}  // namespace py
}  // namespace ora

