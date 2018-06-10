#include <vector>

#include "py_calendar.hh"
#include "py_date.hh"

namespace ora {
namespace py {

//------------------------------------------------------------------------------
// Helpers

namespace {

Range<Date>
parse_range(
  Object* arg)
{
  if (Sequence::Check(arg)) {
    auto seq = cast<Sequence>(arg);
    if (seq->Length() == 2) {
      auto min = convert_to_date(seq->GetItem(0));
      auto max = convert_to_date(seq->GetItem(1));
      if (min <= max)
        return {min, max};
      else
        throw ValueError("range max cannot precede min");
    }
  }

  throw TypeError("not a date range");
}


}  // anonymous namespace

//------------------------------------------------------------------------------

ref<Object>
parse_calendar(
  Module* /* module */,
  Tuple* const args,
  Dict* kw_args)
{
  static char const* const arg_names[] = {"lines", nullptr};
  Object* lines;
  Arg::ParseTupleAndKeywords(args, kw_args, "O", arg_names, &lines);

  LineIter line_iter(lines);
  return PyCalendar::create(ora::parse_calendar(line_iter));
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


Methods<Module>&
add_cal_functions(
  Methods<Module>& methods)
{
  return methods
    .add<parse_calendar>            ("parse_calendar",          nullptr)
    .add<make_weekday_calendar>     ("make_weekday_calendar",   nullptr)
    ;
}


//------------------------------------------------------------------------------

}  // namespace py
}  // namespace ora

