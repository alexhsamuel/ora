#include "cron/ez.hh"
#include "cron/format.hh"
#include "cron/time.hh"
#include "gtest/gtest.h"

using namespace alxs;
using namespace alxs::cron;
using namespace alxs::cron::ez;

//------------------------------------------------------------------------------
// Class Time
//------------------------------------------------------------------------------

TEST(Time, comparisons) {
  Time const t = Time(2013, 6, 29, 18, 27, 13, *get_time_zone("US/Eastern"));
  Time const i = Time::INVALID;
  Time const m = Time::MISSING;

  EXPECT_FALSE(t <  t);
  EXPECT_FALSE(t <  i);
  EXPECT_FALSE(t <  m);
  EXPECT_FALSE(i <  t);
  EXPECT_FALSE(i <  i);
  EXPECT_FALSE(i <  m);
  EXPECT_FALSE(m <  t);
  EXPECT_FALSE(m <  i);
  EXPECT_FALSE(m <  m);

  EXPECT_TRUE (t == t);
  EXPECT_FALSE(t == i);
  EXPECT_FALSE(t == m);
  EXPECT_FALSE(i == t);
  EXPECT_FALSE(i == i);
  EXPECT_FALSE(i == m);
  EXPECT_FALSE(m == t);
  EXPECT_FALSE(m == i);
  EXPECT_FALSE(m == m);

  EXPECT_TRUE (t >= t);
  EXPECT_FALSE(t >= i);
  EXPECT_FALSE(t >= m);
  EXPECT_FALSE(i >= t);
  EXPECT_FALSE(i >= i);
  EXPECT_FALSE(i >= m);
  EXPECT_FALSE(m >= t);
  EXPECT_FALSE(m >= i);
  EXPECT_FALSE(m >= m);

  EXPECT_TRUE (t == Time(2013, 6, 29, 18, 27, 13,   *get_time_zone("US/Eastern")));
  EXPECT_TRUE (t >  Time(2013, 6, 28, 18, 27, 13,   *get_time_zone("US/Eastern")));
  EXPECT_TRUE (t <  Time(2013, 6, 29, 18, 27, 13.1, *get_time_zone("US/Eastern")));
  EXPECT_FALSE(t >  Time(2013, 6, 29, 18, 27, 13.1, *get_time_zone("US/Eastern")));
}

TEST(Time, from_parts) {
  Daytime const daytime = Daytime(18, 27, 13);
  auto const tz = get_time_zone("US/Eastern");
  Unix32Time const time0(2013/JUL/29, daytime, *tz);
  EXPECT_EQ(1375136833, time0.get_offset());

  Time::Offset const offset = 4262126704878682112l;
  Time const time1 = Time::from_offset(offset);
  EXPECT_EQ(offset, time1.get_offset());
  TimeParts const parts1 = time1.get_parts("US/Eastern");
  EXPECT_EQ(2013, parts1.date.year);
  EXPECT_EQ(6, parts1.date.month);
  EXPECT_EQ(27, parts1.date.day);

  Time const time2(2013/JUL/28, Daytime(15, 37, 38), *tz);
  EXPECT_EQ(offset, time2.get_offset());

  Time const time3(2013, 6, 27, 15, 37, 38, *tz);
  EXPECT_EQ(offset, time3.get_offset());
}

TEST(Time, from_parts_dst) {
  auto const tz = get_time_zone("US/Eastern");

  // Test transition to DST.
  Date const dst0 = 2013/MAR/10;
  EXPECT_EQ(Time(dst0, Daytime(6, 59, 0), *UTC), Time(dst0, Daytime(1, 59, 0), *tz));
  EXPECT_EQ(Time(dst0, Daytime(7,  0, 0), *UTC), Time(dst0, Daytime(3,  0, 0), *tz));
  EXPECT_EQ(Time(dst0, Daytime(7,  0, 0), *UTC), Time(dst0, Daytime(3,  0, 0), *tz, false));

  // Test transition from DST.
  Date const dst1 = 2013/NOV/3;
  EXPECT_EQ(Time(dst1, Daytime(4, 59, 0), *UTC), Time(dst1, Daytime(0, 59, 0), *tz));
  EXPECT_EQ(Time(dst1, Daytime(5,  0, 0), *UTC), Time(dst1, Daytime(1,  0, 0), *tz));
  EXPECT_EQ(Time(dst1, Daytime(5,  0, 0), *UTC), Time(dst1, Daytime(1,  0, 0), *tz, true));
  EXPECT_EQ(Time(dst1, Daytime(5, 59, 0), *UTC), Time(dst1, Daytime(1, 59, 0), *tz));
  EXPECT_EQ(Time(dst1, Daytime(5, 59, 0), *UTC), Time(dst1, Daytime(1, 59, 0), *tz, true));
  EXPECT_EQ(Time(dst1, Daytime(6,  0, 0), *UTC), Time(dst1, Daytime(1,  0, 0), *tz, false));
  EXPECT_EQ(Time(dst1, Daytime(6, 59, 0), *UTC), Time(dst1, Daytime(1, 59, 0), *tz, false));
  EXPECT_EQ(Time(dst1, Daytime(7,  0, 0), *UTC), Time(dst1, Daytime(2,  0, 0), *tz));
  EXPECT_EQ(Time(dst1, Daytime(7,  0, 0), *UTC), Time(dst1, Daytime(2,  0, 0), *tz, false));
}

TEST(Time, from_parts_invalid) {
  auto const tz = get_time_zone("US/Eastern");

  EXPECT_THROW(Time(Date::INVALID, Daytime(0, 0, 0), *tz), InvalidDateError);
  EXPECT_THROW(Time(Date::MISSING, Daytime(0, 0, 0), *tz), InvalidDateError);

  EXPECT_TRUE(Time(2013/JUL/28, Daytime::INVALID, *tz).is_invalid());
  EXPECT_TRUE(Time(2013/JUL/28, Daytime(24, 0, 0), *tz).is_invalid());
  EXPECT_TRUE(Time(2013/JUL/28, Daytime(0, 60, 0), *tz).is_invalid());
  EXPECT_TRUE(Time(2013/JUL/28, Daytime(0, 0, 60), *tz).is_invalid());

  EXPECT_TRUE(Time(10000,  0,  0,  0,  0,  0, *tz).is_invalid());
  EXPECT_TRUE(Time( 2013, 12,  0,  0,  0,  0, *tz).is_invalid());
  EXPECT_TRUE(Time( 2013,  0, 31,  0,  0,  0, *tz).is_invalid());
  EXPECT_TRUE(Time( 2013,  0,  0, 24,  0,  0, *tz).is_invalid());
  EXPECT_TRUE(Time( 2013,  0,  0,  0, 60,  0, *tz).is_invalid());
  EXPECT_TRUE(Time( 2013,  0,  0,  0,  0, 60, *tz).is_invalid());

  EXPECT_TRUE(Time( 2013,  2,  9,  1, 59, 59, *tz).is_valid());
  EXPECT_TRUE(Time( 2013,  2,  9,  2,  0,  0, *tz).is_invalid());
  EXPECT_TRUE(Time( 2013,  2,  9,  2, 59, 59, *tz).is_invalid());
  EXPECT_TRUE(Time( 2013,  2,  9,  3,  0,  0, *tz).is_valid());
  EXPECT_TRUE(Time( 2013,  2,  9,  3,  0,  0, *tz, false).is_valid());
}

TEST(Time, get_parts) {
  // 2013 July 28 15:37:38.125 EDT [UTC-4].
  Time const time = Time::from_offset(4262126704887070720l);
  EXPECT_EQ(1375040258, Unix64Time(time).get_offset());

  auto const time_zone = get_time_zone("US/Eastern");
  TimeParts const parts = time.get_parts(*time_zone);
  EXPECT_EQ(2013,       parts.date.year);
  EXPECT_EQ(6,          parts.date.month);
  EXPECT_EQ(27,         parts.date.day);
  EXPECT_EQ(15,         parts.daytime.hour);
  EXPECT_EQ(37,         parts.daytime.minute);
  EXPECT_EQ(38.125,     parts.daytime.second);
  EXPECT_EQ(-14400,     parts.time_zone.offset);
  EXPECT_EQ(true,       parts.time_zone.is_dst);
  EXPECT_STREQ("EDT",   parts.time_zone.abbreviation);
}

TEST(Time, get_parts_invalid) {
  TimeParts const parts = Time::INVALID.get_parts();
  EXPECT_FALSE(year_is_valid(parts.date.year));
  EXPECT_FALSE(month_is_valid(parts.date.month));
  EXPECT_FALSE(day_is_valid(parts.date.day));
  EXPECT_FALSE(hour_is_valid(parts.daytime.hour));
  EXPECT_FALSE(minute_is_valid(parts.daytime.minute));
  EXPECT_FALSE(second_is_valid(parts.daytime.second));
  EXPECT_FALSE(time_zone_offset_is_valid(parts.time_zone.offset));
}

TEST(Time, default_format) {
  EXPECT_EQ("%Y-%m-%d %H:%M:%S %~Z", TimeFormat::get_default().get_pattern());
}

TEST(Time, ostream) {
  auto const tz = get_time_zone("US/Eastern");
  Time const time(2013/JUL/29, Daytime(18, 27, 13.6316313), *tz);
  set_display_time_zone(tz);

  {
    std::stringstream ss;
    ss << time;
    EXPECT_EQ("2013-07-29 18:27:14 EDT", ss.str());
  }

  {
    std::stringstream ss;
    ss << TimeFormat::ISO_LOCAL_BASIC(time);
    EXPECT_EQ("20130729T182714", ss.str());
  }
  
  {
    std::stringstream ss;
    ss << time;
    EXPECT_EQ("2013-07-29 18:27:14 EDT", ss.str());
  }
}

TEST(Time, to_string) {
  auto const tz = get_time_zone("US/Eastern");
  Time const time(2013/JUL/29, Daytime(18, 27, 13.6316313), *tz);

  EXPECT_EQ("2013-07-29 18:27:14 EDT", to_string(time));
  EXPECT_EQ("INVALID                ", to_string(Time::INVALID));
  EXPECT_EQ("MISSING                ", to_string(Time::MISSING));

  EXPECT_EQ("2013-07-29T18:27:14-04:00", TimeFormat::ISO_ZONE_EXTENDED(time));
  EXPECT_EQ("INVALID                  ", TimeFormat::ISO_ZONE_EXTENDED(Time::INVALID));
  EXPECT_EQ("MISSING                  ", TimeFormat::ISO_ZONE_EXTENDED(Time::MISSING));
}

//------------------------------------------------------------------------------
// Class Unix32Time.

TEST(Unix32Time, zero) {
  auto const time = Unix32Time::from_offset(1374522232);

  auto const date = time.get_utc_date<Date>();
  auto const date_parts = date.get_parts();
  EXPECT_EQ(2013, date_parts.year);
  EXPECT_EQ(6, date_parts.month);
  EXPECT_EQ(21, date_parts.day);

  auto const daytime = time.get_utc_daytime<Daytime>();
  auto const daytime_parts = daytime.get_parts();
  EXPECT_EQ(19, daytime_parts.hour);
  EXPECT_EQ(43, daytime_parts.minute);
  EXPECT_EQ(52.0, daytime_parts.second);
}

