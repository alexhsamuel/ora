#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "aslib/string.hh"
#include "cron/calendar.hh"
#include "cron/date_nex.hh"

namespace cron {

using namespace aslib;
using date::Date;

//------------------------------------------------------------------------------
// Helper functions.
//------------------------------------------------------------------------------

// FIXME: We need a unified parsing library.

namespace {

inline std::string
get_line(
  std::istream& in,
  char delim='\n')
{
  std::string line;
  char buffer[256];
  do {
    in.getline(buffer, sizeof(buffer), delim);
    line += buffer;
    // Keep going, if no delimiter was found in this buffer-full.
  } while (! in.eof() && in.failbit && in.gcount() == sizeof(buffer));
  return line;
}


}  // anonymous namespace

//------------------------------------------------------------------------------
// Functions
//------------------------------------------------------------------------------

std::unique_ptr<HolidayCalendar>
parse_holiday_calendar(
  std::istream& in)
{
  std::vector<Date> dates;
  Date min = Date::MISSING;
  Date max = Date::MISSING;
  Date date_min = Date::MISSING;
  Date date_max = Date::MISSING;

  while (!in.eof()) {
    std::string line = strip(get_line(in));
    // Skip blank and comment lines.
    if (line.size() == 0 || line[0] == '#')
      continue;
    StringPair parts = split1(line);
    // FIXME: Handle exceptions.
    if (parts.first == "MIN") 
      min = date::from_iso_date<Date>(parts.second);
    else if (parts.first == "MAX")
      max = date::from_iso_date<Date>(parts.second);
    else {
      auto const date = date::from_iso_date<Date>(parts.first);
      dates.push_back(date);
      // Keep track of the min and max dates we've seen.
      if (!date::nex::before(date_min, date))
        date_min = date;
      if (!date::nex::before(date, date_max))
        date_max = date + 1;
    }
  }

  // Infer missing min or max from the range of given dates.
  if (min.is_missing()) 
    min = dates.size() > 0 ? date_min : Date::MIN;
  assert(!min.is_missing());
  if (max.is_missing()) 
    max = dates.size() > 0 ? date_max : Date::MIN;
  assert(!max.is_missing());

  // Now construct the calendar.
  assert(min <= max);
  auto cal = std::make_unique<HolidayCalendar>(min, max);
  for (auto const date : dates)
    cal->add(date);
  return std::move(cal);
}


std::unique_ptr<HolidayCalendar>
load_holiday_calendar(
  fs::Filename const& filename)
{
  std::ifstream in((char const*) filename);
  return parse_holiday_calendar(in);
}


//------------------------------------------------------------------------------

}  // namespace cron


