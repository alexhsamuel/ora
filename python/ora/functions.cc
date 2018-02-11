#include <cassert>
#include <cstring>
#include <time.h>

#include "ora.hh"
#include "functions_doc.hh"
#include "py.hh"
#include "PyDate.hh"
#include "PyTime.hh"
#include "util.hh"

//------------------------------------------------------------------------------

namespace ora {
namespace py {

namespace {

//------------------------------------------------------------------------------

ref<Object>
days_in_month(
  Module* /* module */,
  Tuple* const args,
  Dict* const kw_args)
{
  static char const* arg_names[] = {"year", "month", nullptr};
  ora::Year year;
  ora::Month month;
  static_assert(
    sizeof(ora::Year) == sizeof(unsigned short), "wrong type for year");
  static_assert(
    sizeof(ora::Month) == sizeof(unsigned char), "wrong type for month");
  Arg::ParseTupleAndKeywords(args, kw_args, "Hb", arg_names, &year, &month);

  if (ora::year_is_valid(year) && ora::month_is_valid(month))
    return Long::FromLong(ora::days_in_month(year, month));
  else
    throw Exception(PyExc_ValueError, "invalid year");
}


ref<Object>
days_in_year(
  Module* /* module */,
  Tuple* const args,
  Dict* const kw_args)
{
  static char const* arg_names[] = {"year", nullptr};
  ora::Year year;
  static_assert(
    sizeof(ora::Year) == sizeof(unsigned short), "wrong type for year");
  Arg::ParseTupleAndKeywords(args, kw_args, "H", arg_names, &year);

  if (ora::year_is_valid(year))
    return Long::FromLong(ora::days_in_year(year));
  else
    throw ValueError("invalid year");
}


ref<Object>
format_iso(
  Module* /* module */,
  Tuple* const args,
  Dict* const kw_args)
{
  static char const* arg_names[] = {"time", "time_zone", "precision", nullptr};
  Object* time_arg;
  Object* tz_arg = nullptr;
  int precision = -1;
  Arg::ParseTupleAndKeywords(
    args, kw_args, "O|Oi", arg_names, &time_arg, &tz_arg, &precision);

  auto api = PyTimeAPI::get(time_arg);
  if (api == nullptr)
    throw TypeError("not a Time");
  auto const tz = tz_arg == nullptr ? UTC : convert_to_time_zone(tz_arg);

  auto ldd = api->to_local_datenum_daytick(time_arg, *tz);
  StringBuilder sb;
  time::format_iso_time(
    sb, datenum_to_ymd(ldd.datenum), daytick_to_hms(ldd.daytick), 
    ldd.time_zone, precision);
  return Unicode::from(sb.str());
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
    throw TypeError("not a type: "s + *time_type->Repr());
  auto const api = PyTimeAPI::get((PyTypeObject*) time_type);
  if (api == nullptr)
    throw TypeError("not a time type: "s + *time_type->Repr());

  auto const datenum    = to_datenum(date_arg);
  auto const daytick    = to_daytick(daytime_arg);
  auto const tz         = convert_to_time_zone(tz_arg);
  return api->from_local_datenum_daytick(datenum, daytick, *tz, first);
}


ref<Object>
get_display_time_zone(
  Module* /* module */,
  Tuple* const args,
  Dict* const kw_args)
{
  static char const* arg_names[] = {nullptr};
  Arg::ParseTupleAndKeywords(args, kw_args, "", arg_names);

  return PyTimeZone::create(ora::get_display_time_zone());
}


ref<Object>
get_system_time_zone(
  Module* /* module */,
  Tuple* const args,
  Dict* const kw_args)
{
  static char const* arg_names[] = {nullptr};
  Arg::ParseTupleAndKeywords(args, kw_args, "", arg_names);

  return PyTimeZone::create(ora::get_system_time_zone());
}


ref<Object>
get_zoneinfo_dir(
  Module* /* module */,
  Tuple* const args,
  Dict* const kw_args)
{
  static char const* arg_names[] = {nullptr};
  Arg::ParseTupleAndKeywords(args, kw_args, "", arg_names);

  return Unicode::from(ora::get_zoneinfo_dir());
}


ref<Object>
is_leap_year(
  Module* /* module */,
  Tuple* const args,
  Dict* const kw_args)
{
  static char const* arg_names[] = {"year", nullptr};
  ora::Year year;
  static_assert(
    sizeof(ora::Year) == sizeof(unsigned short), "wrong type for year");
  Arg::ParseTupleAndKeywords(args, kw_args, "H", arg_names, &year);

  if (ora::year_is_valid(year))
    return Bool::from(ora::is_leap_year(year));
  else
    throw ValueError("invalid year");
}


ref<Object>
now(
  Module* /* module */,
  Tuple* const args,
  Dict* const kw_args)
{
  static char const* arg_names[] = {"Time", nullptr};
  PyTypeObject* time_type = &PyTimeDefault::type_;
  // A bit hacky, but we don't check that time_type is a type object because
  // PyTimeAPI won't accept anything else.
  Arg::ParseTupleAndKeywords(args, kw_args, "|O", arg_names, &time_type);

  auto const api = PyTimeAPI::get(time_type);
  if (api == nullptr)
    throw TypeError("not a time type");
  else
    return api->now();
}


ref<Object>
parse_date(
  Module* /* module */,
  Tuple* const args,
  Dict* const kw_args)
{
  static char const* arg_names[] = {"pattern", "string", nullptr};
  char const* pattern;
  char const* string;
  Arg::ParseTupleAndKeywords(args, kw_args, "ss", arg_names, &pattern, &string);

  // FIXME: Support other date types.
  auto const date = ora::date::parse<Date>(pattern, string);
  if (date.is_valid())
    return PyDate<Date>::create(date);
  else
    // FIXME
    throw ValueError("parse error");
}


ref<Object>
set_display_time_zone(
  Module* /* module */,
  Tuple* const args,
  Dict* const kw_args)
{
  static char const* arg_names[] = {"time_zone", nullptr};
  Object* tz_arg;
  Arg::ParseTupleAndKeywords(args, kw_args, "O", arg_names, &tz_arg);

  auto tz = convert_to_time_zone(tz_arg);
  ora::set_display_time_zone(std::move(tz));
  return none_ref();
}


ref<Object>
set_zoneinfo_dir(
  Module* /* module */,
  Tuple* const args,
  Dict* const kw_args)
{
  static char const* arg_names[] = {"path", nullptr};
  char* path;
  Arg::ParseTupleAndKeywords(args, kw_args, "s", arg_names, &path);

  ora::set_zoneinfo_dir(path);
  return none_ref();
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
    : ora::time::to_local_datenum_daytick(
        convert_to_time<ora::time::Time>(time), *tz);
  return PyLocal::create(
    make_date(local.datenum, date_type),
    make_daytime(local.daytick, daytime_type));
}


ref<Object>
today(
  Module* /* module */,
  Tuple* const args,
  Dict* const kw_args)
{
  static char const* const arg_names[] = {"time_zone", "Date", nullptr};
  Object* tz;
  PyTypeObject* date_type = &PyDateDefault::type_;
  // A bit hacky, but we don't check that date_type is a type object because
  // PyDateAPI won't accept anything else.
  Arg::ParseTupleAndKeywords(
    args, kw_args, "O|O!", arg_names, 
    &tz, &PyType_Type, &date_type);

  auto api = PyDateAPI::get(date_type);
  if (api == nullptr)
    throw TypeError("not a date type");
  else
    return api->today(*convert_to_time_zone(tz));
}


}  // anonymous namespace

//------------------------------------------------------------------------------

Methods<Module>&
add_functions(
  Methods<Module>& methods)
{
  return methods
    .add<days_in_month>             ("days_in_month",           docstring::days_in_month)
    .add<days_in_year>              ("days_in_year",            docstring::days_in_year)
    .add<format_iso>                ("format_iso",              docstring::format_iso)
    .add<from_local>                ("from_local",              docstring::from_local)
    .add<get_display_time_zone>     ("get_display_time_zone",   docstring::get_display_time_zone)
    .add<get_system_time_zone>      ("get_system_time_zone",    docstring::get_system_time_zone)
    .add<get_zoneinfo_dir>          ("get_zoneinfo_dir",        docstring::get_zoneinfo_dir)
    .add<is_leap_year>              ("is_leap_year",            docstring::is_leap_year)
    .add<now>                       ("now",                     docstring::now)
    .add<parse_date>                ("parse_date",              nullptr)  // FIXME
    .add<set_display_time_zone>     ("set_display_time_zone",   docstring::set_display_time_zone)
    .add<set_zoneinfo_dir>          ("set_zoneinfo_dir",        docstring::set_zoneinfo_dir)
    .add<to_local>                  ("to_local",                docstring::to_local)
    .add<today>                     ("today",                   docstring::today)
    ;
}


}  // namespace py
}  // namespace ora

