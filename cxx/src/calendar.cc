#include <algorithm>
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

Calendar
make_const_calendar(
  Interval<Date> const range,
  bool const contains)
{
  auto dates = std::vector<bool>(range.length(), contains);
  return {range.start, std::move(dates)};
}


Calendar
make_weekday_calendar(
  Interval<Date> const range,
  bool const mask[7])
{
  auto dates = std::vector<bool>();
  auto const length = range.length();
  dates.reserve(length);
  for (auto i = 0; i < length; ++i)
    dates.push_back(mask[get_weekday(range.start + i)]);
  return {range.start, std::move(dates)};
}


//------------------------------------------------------------------------------
// Calendar file

/*
  Holiday calendar file format:
    - Line-oriented text, delimited by NL.
    - Leading and trailing whitespace on each line stripped.
    - Blank lines ignored.
    - Lines beginning with # ignored as comment lines.
    - All dates specified as ISO dates, 'YYYY-MM-DD'
    - Range optionally specified with lines 'START <date>' and 'STOP <date>'.
    - Every other line consists of a holiday date followed by whitespace;
      the rest of the line is ignored.
    - If range min or max is not specified, it is inferred from the dates.

  Example:

      # U.S. holidays for the year 2010.

      MIN 2010-01-01
      MAX 2011-01-01

      2010-01-01 New Year's Day
      2010-01-18 Birthday of Martin Luther King, Jr.
      2010-02-15 Washington's Birthday
      2010-05-31 Memorial Day
      2010-07-05 Independence Day
      2010-09-06 Labor Day
      2010-10-11 Columbus Day
      2010-11-11 Veterans Day
      2010-11-25 Thanksgiving Day
      2010-12-24 Christmas Day
      2010-12-31 New Year's Day
*/

Calendar
parse_calendar(
  ora::lib::Iter<std::string>& lines)
{
  std::vector<Date> dates;
  auto range = Interval<Date>{Date::MISSING, Date::MISSING};

  for (auto line_iter = lines.next(); line_iter; line_iter = lines.next()) {
    auto line = strip(*line_iter);
    // Skip blank and comment lines.
    if (line.size() == 0 || line[0] == '#')
      continue;
    StringPair parts = split1(line);
    // FIXME: Handle exceptions.
    if (parts.first == "START") 
      range.start = date::from_iso_date<Date>(parts.second);
    else if (parts.first == "STOP")
      range.stop = date::from_iso_date<Date>(parts.second);
    else
      dates.push_back(date::from_iso_date<Date>(parts.first));
  }

  // Infer missing min or max from the range of given dates.
  if (range.start.is_missing())
    range.start = 
      dates.size() == 0 ? Date::MIN 
      : *std::min_element(dates.begin(), dates.end());
  if (range.stop.is_missing())
    range.stop =
      dates.size() == 0 ? Date::MAX
      : *std::max_element(dates.begin(), dates.end()) + 1;

  // FIXME: Exceptions instead.
  assert(!range.start.is_missing());
  assert(!range.stop.is_missing());
  assert(range.start <= range.stop);

  // Now construct the calendar.
  return {range, dates};
}


Calendar
load_calendar(
  fs::Filename const& filename)
{
  auto in = std::ifstream((const char*) filename);
  auto lines = ora::lib::fs::LineIter(in);
  return parse_calendar(lines);
}


//------------------------------------------------------------------------------

}  // namespace ora


