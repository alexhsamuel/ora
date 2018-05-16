#include <vector>

#include "py_calendar.hh"
#include "py_date.hh"

//------------------------------------------------------------------------------

namespace ora {
namespace py {

ref<Object>
weekday_calendar(
  Module* /* module */,
  Tuple* const args,
  Dict* kw_args)
{
  static char const* const arg_names[] = {"weekdays", nullptr};
  Object* weekdays_arg;
  Arg::ParseTupleAndKeywords(args, kw_args, "O", arg_names, &weekdays_arg);

  auto const weekdays_iter = weekdays_arg->GetIter();
  std::vector<Weekday> weekdays;
  while (auto weekday = weekdays_iter->Next())
    weekdays.push_back(convert_to_weekday(weekday));

  return PyCalendar::create(std::make_shared<WeekdaysCalendar>(weekdays));
}


Methods<Module>&
add_cal_functions(
  Methods<Module>& methods)
{
  return methods
    .add<weekday_calendar>          ("weekday_calendar",        nullptr)
    ;
}


//------------------------------------------------------------------------------

}  // namespace py
}  // namespace ora

