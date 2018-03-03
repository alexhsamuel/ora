#include "ora.hh"
#include "gtest/gtest.h"

using namespace ora::lib;
using namespace ora;
using namespace ora::ez;

//------------------------------------------------------------------------------

TEST(Time, ostream) {
  auto const tz = get_time_zone("US/Eastern");
  auto const time = from_local(2013/JUL/29, from_hms(18, 27, 13.6316313), *tz);
  set_display_time_zone(tz);

  {
    std::stringstream ss;
    ss << time;
    EXPECT_EQ("2013-07-29T22:27:13+00:00", ss.str());
  }

  {
    std::stringstream ss;
    ss << TimeFormat::ISO_LOCAL_BASIC(time, DTZ);
    EXPECT_EQ("20130729T182713", ss.str());
  }
  
  {
    std::stringstream ss;
    ss << time;
    EXPECT_EQ("2013-07-29T22:27:13+00:00", ss.str());
  }
}

TEST(Time, to_string) {
  auto const tz = get_time_zone("US/Eastern");
  auto const time = from_local(2013/JUL/29, from_hms(18, 27, 13.6316313), *tz);

  EXPECT_EQ("2013-07-29T22:27:13+00:00", to_string(time));
  EXPECT_EQ("INVALID", to_string(Time::INVALID));
  EXPECT_EQ("MISSING", to_string(Time::MISSING));
}

TEST(Time, to_string_range) {
  EXPECT_EQ("0001-01-01T00:00:00+00:00", to_string(Time::MIN));
  EXPECT_EQ("9999-12-31T23:59:59+00:00", to_string(Time::MAX));
  EXPECT_EQ("1970-01-01T00:00:00+00:00", to_string(SmallTime::MIN));
  EXPECT_EQ("2106-02-07T06:28:13+00:00", to_string(SmallTime::MAX));
  EXPECT_EQ("1677-09-21T00:12:43+00:00", to_string(NsTime::MIN));
  EXPECT_EQ("2262-04-11T23:47:16+00:00", to_string(NsTime::MAX));
  EXPECT_EQ("1901-12-13T20:45:52+00:00", to_string(Unix32Time::MIN));
  EXPECT_EQ("2038-01-19T03:14:05+00:00", to_string(Unix32Time::MAX));
  EXPECT_EQ("0001-01-01T00:00:00+00:00", to_string(Unix64Time::MIN));
  EXPECT_EQ("9999-12-31T23:59:59+00:00", to_string(Unix64Time::MAX));
  EXPECT_EQ("0001-01-01T00:00:00+00:00", to_string(Time128::MIN));
  EXPECT_EQ("9999-12-31T23:59:59+00:00", to_string(Time128::MAX));
}

TEST(Time, standard_formats) {
  auto const tz = get_time_zone("US/Eastern");
  auto const time = from_local(2013/JUL/29, from_hms(18, 27, 13.6316313), *tz);

  EXPECT_EQ("2013-07-29T18:27:13-04:00", TimeFormat::ISO_ZONE_EXTENDED(time, *tz));
  EXPECT_EQ("INVALID                  ", TimeFormat::ISO_ZONE_EXTENDED(Time::INVALID, *tz));
  EXPECT_EQ("MISSING                  ", TimeFormat::ISO_ZONE_EXTENDED(Time::MISSING, *tz));
}

