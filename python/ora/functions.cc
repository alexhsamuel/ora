#include <cassert>
#include <cstring>
#include <time.h>

#include "ora.hh"
#include "py.hh"
#include "PyDate.hh"
#include "PyTime.hh"
#include "util.hh"

using namespace std::string_literals;

//------------------------------------------------------------------------------

namespace ora {
namespace py {

//------------------------------------------------------------------------------
// Docstrings
//------------------------------------------------------------------------------

namespace docstring {

using doct_t = char const* const;

#include "functions.docstrings.hh.inc"
#include "functions.docstrings.cc.inc"

}  // namespace docstring

//------------------------------------------------------------------------------

namespace {

Exception
parse_error(
  size_t const string_pos)
{
  static auto exc_type = import("ora", "ParseError");
  return Exception(
    exc_type, "parse error at pos "s + to_string<int>(string_pos));
}


Exception
parse_error(
  size_t const pattern_pos,
  size_t const string_pos)
{
  static auto exc_type = import("ora", "ParseError");
  return Exception(
    exc_type, 
      "parse error at pattern pos "s + to_string<int>(pattern_pos)
    + ", string pos " + to_string<int>(string_pos));
}


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
format_time(
  Module* /* module */,
  Tuple* const args,
  Dict* const kw_args)
{
  static char const* const arg_names[] 
    = {"pattern", "time", "time_zone", nullptr};
  Object* time_arg;
  char* pattern;
  Object* time_zone = nullptr;
  Arg::ParseTupleAndKeywords(
    args, kw_args, "sO|O", arg_names, &pattern, &time_arg, &time_zone);

  auto api = PyTimeAPI::get(time_arg);
  if (api == nullptr)
    throw TypeError("not a Time");
  auto const tz = 
    time_zone == nullptr ? ora::UTC : convert_to_time_zone(time_zone);

  // FIXME: Need to convert to LocalDatenumDayticks to format, but must handle
  // invalid and missing dates first.
  ora::time::TimeFormat const fmt(pattern);
  return Unicode::from(
      api->is_invalid(time_arg) ? fmt.get_invalid()
    : api->is_missing(time_arg) ? fmt.get_missing()
    : fmt(api->to_local_datenum_daytick(time_arg, *tz)));
  // return Unicode::from(ora::time::TimeFormat(pattern)(->time_, *tz));
}


ref<Object>
format_time_iso(
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
  static char const* arg_names[] = {"pattern", "string", "Date", nullptr};
  char const* pattern;
  char const* string;
  PyTypeObject* date_type = &PyDateDefault::type_;
  Arg::ParseTupleAndKeywords(
    args, kw_args, "ss|$O", arg_names, &pattern, &string, &date_type);

  auto const api = PyDateAPI::get(date_type);
  if (api == nullptr)
    throw TypeError("not a date type");

  FullDate parts;
  char const* p = pattern;
  char const* s = string;
  if (ora::date::parse_date_parts(p, s, parts))
    return api->from_parts(parts);
  else
    throw parse_error(p - pattern, s - string);
} 


ref<Object>
parse_daytime(
  Module* /* module */,
  Tuple* const args,
  Dict* const kw_args)
{
  static char const* arg_names[] = {"pattern", "string", "Daytime", nullptr};
  char const* pattern;
  char const* string;
  PyTypeObject* daytime_type = &PyDaytimeDefault::type_;
  Arg::ParseTupleAndKeywords(
    args, kw_args, "ss|$O", arg_names, &pattern, &string, &daytime_type);

  auto const api = PyDaytimeAPI::get(daytime_type);
  if (api == nullptr)
    throw TypeError("not a daytime type");

  HmsDaytime parts;
  char const* p = pattern;
  char const* s = string;
  if (ora::daytime::parse_daytime_parts(p, s, parts))
    return api->from_hms(parts);
  else
    throw parse_error(p - pattern, s - string);
} 


ref<Object>
parse_time(
  Module* /* module */,
  Tuple* const args,
  Dict* const kw_args)
{
  static char const* arg_names[] 
    = {"pattern", "string", "time_zone", "first", "Time", nullptr};
  char const* pattern;
  char const* string;
  Object* tz_arg = nullptr;
  int first = true;
  PyTypeObject* time_type = &PyTimeDefault::type_;
  Arg::ParseTupleAndKeywords(
    args, kw_args, "ss|O$pO", arg_names, 
    &pattern, &string, &tz_arg, &first, &time_type);

  auto const api = PyTimeAPI::get(time_type);
  if (api == nullptr)
    throw TypeError("not a time type");

  auto const tz = tz_arg == nullptr ? nullptr : convert_to_time_zone(tz_arg);

  FullDate date;
  HmsDaytime hms;
  ora::time::TimeZoneInfo tz_info;
  char const* p = pattern;
  char const* s = string;
  if (ora::time::parse_time_parts(p, s, date, hms, tz_info)) {
    // FIXME: Factor this logic into a function.
    auto const datenum = parts_to_datenum(date);
    auto const daytick = hms_to_daytick(hms.hour, hms.minute, hms.second);
    if (!tz_info.name.empty()) {
      TimeZone_ptr tz;
      try {
        tz = get_time_zone(tz_info.name);
      }
      catch (ora::lib::ValueError) {
        throw py::ValueError(std::string("not a time zone: ") + tz_info.name);
      }
      return api->from_local_datenum_daytick(datenum, daytick, *tz, first);
    }
    else if (time_zone_offset_is_valid(tz_info.offset))
      return api->from_local_datenum_daytick(datenum, daytick, tz_info.offset);
    else if (tz != nullptr) 
      return api->from_local_datenum_daytick(datenum, daytick, *tz, first);
    else
      throw py::ValueError("no time zone");
  }
  else
    throw parse_error(p - pattern, s - string);
} 


ref<Object>
parse_time_iso(
  Module* /* module */,
  Tuple* const args,
  Dict* const kw_args)
{
  static char const* arg_names[] = {"string", "Time", nullptr};
  char const* string;
  PyTypeObject* time_type = &PyTimeDefault::type_;
  Arg::ParseTupleAndKeywords(
    args, kw_args, "s|$O", arg_names, &string, &time_type);

  auto const api = PyTimeAPI::get(time_type);
  if (api == nullptr)
    throw TypeError("not a time type");

  YmdDate ymd;
  HmsDaytime hms;
  TimeZoneOffset tz_offset;
  char const* s = string;
  if (ora::time::parse_iso_time(s, ymd, hms, tz_offset) && *s == 0) {
    auto const datenum = ymd_to_datenum(ymd.year, ymd.month, ymd.day);
    auto const daytick = hms_to_daytick(hms.hour, hms.minute, hms.second);
    return api->from_local_datenum_daytick(datenum, daytick, tz_offset);
  }
  else
    throw parse_error(s - string);
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
    .add<format_time>               ("format_time",             docstring::format_time)
    .add<format_time_iso>           ("format_time_iso",         docstring::format_time_iso)
    .add<from_local>                ("from_local",              docstring::from_local)
    .add<get_display_time_zone>     ("get_display_time_zone",   docstring::get_display_time_zone)
    .add<get_system_time_zone>      ("get_system_time_zone",    docstring::get_system_time_zone)
    .add<get_zoneinfo_dir>          ("get_zoneinfo_dir",        docstring::get_zoneinfo_dir)
    .add<is_leap_year>              ("is_leap_year",            docstring::is_leap_year)
    .add<now>                       ("now",                     docstring::now)
    .add<parse_date>                ("parse_date",              docstring::parse_date)
    .add<parse_daytime>             ("parse_daytime",           docstring::parse_daytime)
    .add<parse_time>                ("parse_time",              docstring::parse_time)
    .add<parse_time_iso>            ("parse_time_iso",          docstring::parse_time_iso)
    .add<set_display_time_zone>     ("set_display_time_zone",   docstring::set_display_time_zone)
    .add<set_zoneinfo_dir>          ("set_zoneinfo_dir",        docstring::set_zoneinfo_dir)
    .add<to_local>                  ("to_local",                docstring::to_local)
    .add<today>                     ("today",                   docstring::today)
    ;
}


}  // namespace py
}  // namespace ora

