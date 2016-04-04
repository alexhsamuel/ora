#include <cassert>
#include <cstring>
#include <time.h>

#include "cron/date.hh"
#include "cron/time.hh"
#include "cron/time_zone.hh"
#include "cron/types.hh"
#include "py.hh"
#include "PyTime.hh"
#include "util.hh"

//------------------------------------------------------------------------------

using namespace aslib;
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
  Object* time_type = (Object*) &PyTimeDefault::type_;
  Arg::ParseTupleAndKeywords(
    args, kw_args, "(OO)O|pO", arg_names,
    &date_arg, &daytime_arg, &tz_arg, &first, &time_type);

  // Make sure the time type is a PyTime instance, and get its virtual API.
  if (!Type::Check(time_type))
    throw py::TypeError("not a type: "s + *time_type->Repr());
  auto const api = PyTimeAPI::get((PyTypeObject*) time_type);
  if (api == nullptr)
    throw py::TypeError("not a time type: "s + *time_type->Repr());

  auto const datenum    = to_datenum(date_arg);
  auto const daytick    = to_daytick(daytime_arg);
  auto const tz         = convert_to_time_zone(tz_arg);
  return api->from_local_datenum_daytick(datenum, daytick, *tz, first);
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
now(
  Module* /* module */,
  Tuple* const args,
  Dict* const kw_args)
{
  static char const* arg_names[] = {"Time", nullptr};
  PyTypeObject* time_type = &PyTimeDefault::type_;
  Arg::ParseTupleAndKeywords(args, kw_args, "|O", arg_names, &time_type);

  auto const api = PyTimeAPI::get(time_type);
  if (api == nullptr)
    throw TypeError("not a time type");

  return api->now();
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
    return Long::FromLong(cron::days_per_year(year));
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
  PyTypeObject* date_type = &PyDateDefault::type_;
  PyTypeObject* daytime_type = &PyDaytimeDefault::type_;
  static char const* const arg_names[] 
    = {"time", "time_zone", "Date", "Daytime", nullptr};
  Arg::ParseTupleAndKeywords(
    args, kw_args, 
    "OO|O!O!", arg_names, 
    &time, &tz_arg, &PyType_Type, &date_type, &PyType_Type, &daytime_type);
  
  auto const tz = convert_to_time_zone(tz_arg);

  auto api = PyTimeAPI::get(time);
  auto local = 
    // If this is a PyTime object and we have an API, use it.
    api != nullptr ? api->to_local_datenum_daytick(time, *tz)
    // Otherwise, convert to a time and then proceed.
    : cron::to_local_datenum_daytick(convert_to_time<cron::Time>(time), *tz);
  return make_local(local, date_type, daytime_type);
}


ref<Object>
to_local_datenum_daytick(
  Module* /* module */,
  Tuple* const args,
  Dict* const kw_args)
{
  Object* time;
  Object* tz_arg;
  static char const* const arg_names[] = {"time", "time_zone", nullptr};
  Arg::ParseTupleAndKeywords(args, kw_args, "OO", arg_names, &time, &tz_arg);
  
  auto const tz = convert_to_time_zone(tz_arg);
  auto api = PyTimeAPI::get(time);
  auto local = 
    // If this is a PyTime object and we have an API, use it.
    api != nullptr ? api->to_local_datenum_daytick(time, *tz)
    // Otherwise, convert to a time and then proceed.
    : cron::to_local_datenum_daytick(convert_to_time<cron::Time>(time), *tz);
  return make_local_datenum_daytick(local);
}


ref<Object>
today(
  Module* /* module */,
  Tuple* const args,
  Dict* const kw_args)
{
  Object* tz;
  Object* date_type = (Object*) &PyDateDefault::type_;
  static char const* const arg_names[] = {"time_zone", "Date", nullptr};
  Arg::ParseTupleAndKeywords(args, kw_args, "O|O", arg_names, &tz, &date_type);

  auto local = cron::to_local_datenum_daytick(
    cron::now<cron::Time>(), *convert_to_time_zone(tz));
  // FIXME: Use API.
  return date_type
    ->CallMethodObjArgs("from_datenum", Long::from(local.datenum));
}


}  // anonymous namespace

//------------------------------------------------------------------------------

Methods<Module>&
add_functions(
  Methods<Module>& methods)
{
  return methods
    .add<days_per_month>            ("days_per_month")
    .add<from_local>                ("from_local")
    .add<is_leap_year>              ("is_leap_year")
    .add<now>                       ("now")
    .add<ordinals_per_year>         ("ordinals_per_year")
    .add<to_local>                  ("to_local")
    .add<to_local_datenum_daytick>  ("to_local_datenum_daytick")
    .add<today>                     ("today")
    ;
}


