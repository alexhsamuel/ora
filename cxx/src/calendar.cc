#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "ora/lib/file.hh"
#include "ora/lib/string.hh"
#include "ora/calendar.hh"

namespace ora {

using namespace ora::lib;
using date::Date;

//------------------------------------------------------------------------------
// Functions
//------------------------------------------------------------------------------

namespace {

}  // anonymous namespace


Calendar
load_calendar(
  fs::Filename const& filename)
{
  using ora::lib::fs::LineIterator;
  std::ifstream in((char const*) filename);
  return parse_calendar(LineIterator(&in), LineIterator());
}


Calendar
make_const_calendar(
  Range<Date> const range,
  bool const contains)
{
  auto dates = std::vector<bool>(range.max - range.min + 1, contains);
  return {range.min, std::move(dates)};
}


Calendar
make_weekday_calendar(
  Range<Date> const range,
  bool const mask[7])
{
  auto dates = std::vector<bool>();
  auto const length = range.max - range.min + 1;
  dates.reserve(length);
  for (auto i = 0; i < length; ++i)
    dates.push_back(mask[get_weekday(range.min + i)]);
  return {range.min, std::move(dates)};
}


//------------------------------------------------------------------------------

}  // namespace ora


