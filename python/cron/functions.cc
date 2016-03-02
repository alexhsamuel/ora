#include <cassert>

#include "cron/date.hh"
#include "cron/time.hh"
#include "cron/time_zone.hh"
#include "cron/types.hh"
#include "globals.hh"
#include "py.hh"
#include "PyTime.hh"

//------------------------------------------------------------------------------

using namespace alxs;
using namespace py;

namespace {

//------------------------------------------------------------------------------

ref<Object>
days_per_month(
  Module* /* module */,
  Tuple* const args,
  Dict* const kw_args)
{
  static char const* arg_names[] = {"year", "month", nullptr};
  cron::Year year;
  cron::Month month;
  static_assert(
    sizeof(cron::Year) == sizeof(unsigned short), "wrong type for year");
  static_assert(
    sizeof(cron::Month) == sizeof(unsigned char), "wrong type for month");
  Arg::ParseTupleAndKeywords(args, kw_args, "Hb", arg_names, &year, &month);

  --month;
  if (cron::year_is_valid(year) && cron::month_is_valid(month))
    return Long::FromLong(cron::days_per_month(year, month));
  else
    throw Exception(PyExc_ValueError, "invalid year");
}


ref<Object>
from_local(
  Module* /* module */,
  Tuple* const args,
  Dict* const kw_args)
{
  static char const* arg_names[] 
    = {"local_time", "time_zone", "first", "Time", nullptr};
  Object* date_arg;
  Object* daytime_arg;
  Object* tz_arg;
  int first = true;
  // FIXME: DEFAULT_TIME_TYPE constant?
  Object* time_type_arg = (Object*) &PyTime<cron::Time>::type_;
  Arg::ParseTupleAndKeywords(
    args, kw_args, "(OO)O|pO", arg_names,
    &date_arg, &daytime_arg, &tz_arg, &first, &time_type_arg);

  auto const datenum = to_datenum(date_arg);
  auto const daytick = to_daytick(daytime_arg);
  auto const tz      = to_time_zone(tz_arg);

  // Special case fast path for the default time type.
  if (time_type_arg == (Object*) &PyTime<cron::Time>::type_) 
    return PyTime<cron::Time>::create(
      cron::from_local<cron::Time>(datenum, daytick, tz, first));

  else {
    auto factory = time_type_arg->GetAttrString("_from_datenum_daytick");
    if (factory == nullptr)
      throw TypeError("not a time type");
    else {
      // FIXME: Wrap.
      auto result = PyObject_CallFunctionObjArgs(
        (PyObject*) factory,
        (PyObject*) Long::FromLong(datenum),
        (PyObject*) Long::FromLong(daytick),
        (PyObject*) tz_arg,
        (PyObject*) Bool::from(first),
        nullptr);
      check_not_null(result);
      return ref<Object>::take(result);
    }
  }
}


ref<Object>
is_leap_year(
  Module* /* module */,
  Tuple* const args,
  Dict* const kw_args)
{
  static char const* arg_names[] = {"year", nullptr};
  cron::Year year;
  static_assert(
    sizeof(cron::Year) == sizeof(unsigned short), "wrong type for year");
  Arg::ParseTupleAndKeywords(args, kw_args, "H", arg_names, &year);

  if (cron::year_is_valid(year))
    return Bool::from(cron::is_leap_year(year));
  else
    throw py::ValueError("invalid year");
}


ref<Object>
ordinals_per_year(
  Module* /* module */,
  Tuple* const args,
  Dict* const kw_args)
{
  static char const* arg_names[] = {"year", nullptr};
  cron::Year year;
  static_assert(
    sizeof(cron::Year) == sizeof(unsigned short), "wrong type for year");
  Arg::ParseTupleAndKeywords(args, kw_args, "H", arg_names, &year);

  if (cron::year_is_valid(year))
    return Long::FromLong(cron::ordinals_per_year(year));
  else
    throw py::ValueError("invalid year");
}


ref<Object>
to_local(
  Module* /* module */,
  Tuple* const args,
  Dict* const kw_args)
{
  Object* time;
  Object* tz;
  Object* date_type = (Object*) &PyDate<cron::Date>::type_;
  Object* daytime_type = (Object*) &PyDaytime<cron::Daytime>::type_;
  static char const* const arg_names[] 
    = {"time", "time_zone", "Date", "Daytime", nullptr};
  Arg::ParseTupleAndKeywords(
    args, kw_args, 
    "OO|OO", arg_names, 
    &time, &tz, &date_type, &daytime_type);
  
  return make_local(alxs::to_local(time, tz), date_type, daytime_type);
}


}  // anonymous namespace

//------------------------------------------------------------------------------

Methods<Module>&
add_functions(
  Methods<Module>& methods)
{
  return methods
    .add<days_per_month>        ("days_per_month")
    .add<from_local>            ("from_local")
    .add<is_leap_year>          ("is_leap_year")
    .add<ordinals_per_year>     ("ordinals_per_year")
    .add<to_local>              ("to_local")
    ;
}


