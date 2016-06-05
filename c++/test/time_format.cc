#include "cron/ez.hh"
#include "cron/format.hh"
#include "cron/localization.hh"
#include "cron/time.hh"
#include "gtest/gtest.h"

using namespace aslib;
using namespace cron;
using namespace cron::ez;
using namespace cron::time;
using cron::daytime::Daytime;

//------------------------------------------------------------------------------

TEST(Time, ostream) {
  auto const tz = get_time_zone("US/Eastern");
  auto const time = from_local(2013/JUL/29, Daytime(18, 27, 13.6316313), *tz);
  set_display_time_zone(tz);

  {
    std::stringstream ss;
    ss << time;
    EXPECT_EQ("2013-07-29T22:27:14Z", ss.str());
  }

  {
    std::stringstream ss;
    ss << TimeFormat::ISO_LOCAL_BASIC(time, DTZ);
    EXPECT_EQ("20130729T182714", ss.str());
  }
  
  {
    std::stringstream ss;
    ss << time;
    EXPECT_EQ("2013-07-29T22:27:14Z", ss.str());
  }
}

TEST(Time, to_string) {
  auto const tz = get_time_zone("US/Eastern");
  auto const time = from_local(2013/JUL/29, Daytime(18, 27, 13.6316313), *tz);

  EXPECT_EQ("2013-07-29T22:27:14Z", to_string(time));
  EXPECT_EQ("INVALID", to_string(Time::INVALID));
  EXPECT_EQ("MISSING", to_string(Time::MISSING));
}

TEST(Time, standard_formats) {
  auto const tz = get_time_zone("US/Eastern");
  auto const time = from_local(2013/JUL/29, Daytime(18, 27, 13.6316313), *tz);

  EXPECT_EQ("2013-07-29T18:27:14-04:00", TimeFormat::ISO_ZONE_EXTENDED(time, *tz));
  EXPECT_EQ("INVALID                  ", TimeFormat::ISO_ZONE_EXTENDED(Time::INVALID, *tz));
  EXPECT_EQ("MISSING                  ", TimeFormat::ISO_ZONE_EXTENDED(Time::MISSING, *tz));
}

