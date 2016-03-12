#include <cassert>

#include "cron/date.hh"
#include "cron/time.hh"
#include "cron/time_zone.hh"
#include "cron/types.hh"
#include "py.hh"
#include "PyTime.hh"
#include "util.hh"

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
  Object* time_type_arg = (Object*) &PyTimeDefault::type_;
  Arg::ParseTupleAndKeywords(
    args, kw_args, "(OO)O|pO", arg_names,
    &date_arg, &daytime_arg, &tz_arg, &first, &time_type_arg);

  auto const datenum = to_datenum(date_arg);
  auto const daytick = to_daytick(daytime_arg);

  // Special case fast path for the default time type.
  if (time_type_arg == (Object*) &PyTimeDefault::type_) 
    return PyTimeDefault::create(
      cron::from_local<typename PyTimeDefault::Time>(
        datenum, daytick, *convert_to_time_zone(tz_arg), first));

  else {
    auto factory = time_type_arg->GetAttrString("_from_local");
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
  Object* tz_arg;
  Object* date_type = (Object*) &PyDate<cron::Date>::type_;
  Object* daytime_type = (Object*) &PyDaytime<cron::Daytime>::type_;
  static char const* const arg_names[] 
    = {"time", "time_zone", "Date", "Daytime", nullptr};
  Arg::ParseTupleAndKeywords(
    args, kw_args, 
    "OO|OO", arg_names, 
    &time, &tz_arg, &date_type, &daytime_type);
  
  auto const tz = convert_to_time_zone(tz_arg);

  auto api = PyTimeAPI::get(time);
  auto local = 
    // If this is a PyTime object and we have an API, use it.
    api != nullptr ? api->to_local_datenum_daytick(time, *tz)
    // Otherwise, convert to a time and then proceed.
    : cron::to_local_datenum_daytick(convert_to_time<cron::Time>(time), *tz);
  return make_local(local, date_type, daytime_type);
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


