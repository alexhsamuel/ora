#include <cassert>

#include "cron/date.hh"
#include "py.hh"

//------------------------------------------------------------------------------

using namespace alxs;
using namespace py;

namespace {

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
    throw Exception(PyExc_ValueError, "invalid year");
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
    throw Exception(PyExc_ValueError, "invalid year");
}


}  // anonymous namespace

//------------------------------------------------------------------------------

Methods<Module>&
add_functions(
  Methods<Module>& methods)
{
  return methods
    .add<days_per_month>        ("days_per_month")
    .add<is_leap_year>          ("is_leap_year")
    .add<ordinals_per_year>     ("ordinals_per_year")
    ;
}


