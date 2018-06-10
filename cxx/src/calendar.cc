#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "ora/lib/string.hh"
#include "ora/calendar.hh"

namespace ora {

using namespace ora::lib;
using date::Date;

//------------------------------------------------------------------------------
// Functions
//------------------------------------------------------------------------------

// FIXME
// std::unique_ptr<Calendar>
// load_calendar(
//   fs::Filename const& filename)
// {
//   std::ifstream in((char const*) filename);
//   return parse_calendar(in);
// }


std::unique_ptr<Calendar>
make_weekday_calendar(
  Range<Date> const range,
  bool const mask[7])
{
  std::vector<bool> dates;
  auto const length = range.max - range.min;
  dates.reserve(length);
  for (auto i = 0; i < length; ++i)
    dates.push_back(mask[get_weekday(range.min + i)]);
  return std::make_unique<Calendar>(range.min, std::move(dates));
}


//------------------------------------------------------------------------------

}  // namespace ora


