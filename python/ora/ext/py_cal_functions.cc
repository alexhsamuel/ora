#include <vector>

#include "py_calendar.hh"
#include "py_date.hh"

namespace ora {
namespace py {

//------------------------------------------------------------------------------

ref<Object>
make_const_calendar(
  Module* /* module */,
  Tuple* const args,
  Dict* kw_args)
{
  static char const* const arg_names[] = {"range", "contains", nullptr};
  Object* range_arg;
  int contains;
  Arg::ParseTupleAndKeywords(
    args, kw_args, "Op", arg_names, &range_arg, &contains);
  auto const range = parse_range(range_arg);

  return PyCalendar::create(make_const_calendar(range, contains != 0));
}


ref<Object>
make_weekday_calendar(
  Module* /* module */,
  Tuple* const args,
  Dict* kw_args)
{
  static char const* const arg_names[] = {"range", "weekdays", nullptr};
  Object* range_arg;
  Object* weekdays_arg;
  Arg::ParseTupleAndKeywords(
    args, kw_args, "OO", arg_names, &range_arg, &weekdays_arg);
  auto const range = parse_range(range_arg);

  auto const weekdays_iter = weekdays_arg->GetIter();
  bool mask[7] = {false, false, false, false, false, false, false};
  while (auto weekday = weekdays_iter->Next())
    mask[convert_to_weekday(weekday)] = true;

  return PyCalendar::create(make_weekday_calendar(range, mask));
}


ref<Object>
parse_calendar(
  Module* /* module */,
  Tuple* const args,
  Dict* kw_args)
{
  static char const* const arg_names[] = {"lines", nullptr};
  Object* lines;
  Arg::ParseTupleAndKeywords(args, kw_args, "O", arg_names, &lines);

  auto line_iter = LineIter(lines);
  return PyCalendar::create(ora::parse_calendar(line_iter));
}


Methods<Module>&
add_cal_functions(
  Methods<Module>& methods)
{
  // FIXME: Docstrings.
  return methods
    .add<make_const_calendar>       ("make_const_calendar",     nullptr)
    .add<make_weekday_calendar>     ("make_weekday_calendar",   nullptr)
    .add<parse_calendar>            ("parse_calendar",          nullptr)
    ;
}


//------------------------------------------------------------------------------

}  // namespace py
}  // namespace ora

