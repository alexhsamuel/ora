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
// Class Time
//------------------------------------------------------------------------------

TEST(Time, comparisons) {
  Time const t = from_local(2013, 6, 29, 18, 27, 13, "US/Eastern");
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

  EXPECT_TRUE (t == from_local(2013, 6, 29, 18, 27, 13,   "US/Eastern"));
  EXPECT_TRUE (t >  from_local(2013, 6, 28, 18, 27, 13,   "US/Eastern"));
  EXPECT_TRUE (t <  from_local(2013, 6, 29, 18, 27, 13.1, "US/Eastern"));
  EXPECT_FALSE(t >  from_local(2013, 6, 29, 18, 27, 13.1, "US/Eastern"));
}

TEST(Time, from_parts) {
  Daytime const daytime(18, 27, 13);
  auto const tz = get_time_zone("US/Eastern");
  auto const time0 = from_local<Unix32Time>(2013/JUL/29, daytime, *tz);
  EXPECT_EQ(1375136833, time0.get_offset());

  Time::Offset const offset = 4262126704878682112l;
  Time const time1 = Time::from_offset(offset);
  EXPECT_EQ(offset, time1.get_offset());
  TimeParts const parts1 = get_parts(time1, "US/Eastern");
  EXPECT_EQ(2013, parts1.date.year);
  EXPECT_EQ(6, parts1.date.month);
  EXPECT_EQ(27, parts1.date.day);

  auto const time2 = from_local(2013/JUL/28, Daytime(15, 37, 38), *tz);
  EXPECT_EQ(offset, time2.get_offset());

  auto const time3 = from_local(2013, 6, 27, 15, 37, 38, *tz);
  EXPECT_EQ(offset, time3.get_offset());
}

TEST(Time, from_parts_dst) {
  auto const tz = get_time_zone("US/Eastern");

  // Test transition to DST.
  Date const dst0 = 2013/MAR/10;
  EXPECT_EQ(from_local(dst0, Daytime(6, 59, 0), UTC), from_local(dst0, Daytime(1, 59, 0), *tz));
  EXPECT_EQ(from_local(dst0, Daytime(7,  0, 0), UTC), from_local(dst0, Daytime(3,  0, 0), *tz));
  EXPECT_EQ(from_local(dst0, Daytime(7,  0, 0), UTC), from_local(dst0, Daytime(3,  0, 0), *tz, false));

  // Test transition from DST.
  Date const dst1 = 2013/NOV/3;
  EXPECT_EQ(from_local(dst1, Daytime(4, 59, 0), UTC), from_local(dst1, Daytime(0, 59, 0), *tz));
  EXPECT_EQ(from_local(dst1, Daytime(5,  0, 0), UTC), from_local(dst1, Daytime(1,  0, 0), *tz));
  EXPECT_EQ(from_local(dst1, Daytime(5,  0, 0), UTC), from_local(dst1, Daytime(1,  0, 0), *tz, true));
  EXPECT_EQ(from_local(dst1, Daytime(5, 59, 0), UTC), from_local(dst1, Daytime(1, 59, 0), *tz));
  EXPECT_EQ(from_local(dst1, Daytime(5, 59, 0), UTC), from_local(dst1, Daytime(1, 59, 0), *tz, true));
  EXPECT_EQ(from_local(dst1, Daytime(6,  0, 0), UTC), from_local(dst1, Daytime(1,  0, 0), *tz, false));
  EXPECT_EQ(from_local(dst1, Daytime(6, 59, 0), UTC), from_local(dst1, Daytime(1, 59, 0), *tz, false));
  EXPECT_EQ(from_local(dst1, Daytime(7,  0, 0), UTC), from_local(dst1, Daytime(2,  0, 0), *tz));
  EXPECT_EQ(from_local(dst1, Daytime(7,  0, 0), UTC), from_local(dst1, Daytime(2,  0, 0), *tz, false));
}

TEST(Time, from_parts_invalid) {
  auto const tz = get_time_zone("US/Eastern");

  EXPECT_THROW(from_local(Date::INVALID, Daytime(0, 0, 0), *tz), InvalidDateError);
  EXPECT_THROW(from_local(Date::MISSING, Daytime(0, 0, 0), *tz), InvalidDateError);

  EXPECT_THROW(from_local(2013/JUL/28, Daytime::INVALID , *tz), InvalidDaytimeError);
  EXPECT_THROW(from_local(2013/JUL/28, Daytime(24, 0, 0), *tz), InvalidDaytimeError);
  EXPECT_THROW(from_local(2013/JUL/28, Daytime(0, 60, 0), *tz), InvalidDaytimeError);
  EXPECT_THROW(from_local(2013/JUL/28, Daytime(0, 0, 60), *tz), InvalidDaytimeError);

  EXPECT_THROW(from_local(10000,  0,  0,  0,  0,  0, *tz), InvalidDateError);
  EXPECT_THROW(from_local( 2013, 12,  0,  0,  0,  0, *tz), InvalidDateError);
  EXPECT_THROW(from_local( 2013,  0, 31,  0,  0,  0, *tz), InvalidDateError);
  EXPECT_THROW(from_local( 2013,  0,  0, 24,  0,  0, *tz), InvalidDaytimeError);
  EXPECT_THROW(from_local( 2013,  0,  0,  0, 60,  0, *tz), InvalidDaytimeError);
  EXPECT_THROW(from_local( 2013,  0,  0,  0,  0, 60, *tz), InvalidDaytimeError);

  EXPECT_TRUE (from_local( 2013,  2,  9,  1, 59, 59, *tz).is_valid());
  EXPECT_THROW(from_local( 2013,  2,  9,  2,  0,  0, *tz), NonexistentDateDaytime);
  EXPECT_THROW(from_local( 2013,  2,  9,  2, 59, 59, *tz), NonexistentDateDaytime);
  EXPECT_TRUE (from_local( 2013,  2,  9,  3,  0,  0, *tz).is_valid());
  EXPECT_TRUE (from_local( 2013,  2,  9,  3,  0,  0, *tz, false).is_valid());
}

TEST(Time, get_parts) {
  // 2013 July 28 15:37:38.125 EDT [UTC-4].
  auto const time = Time::from_offset(4262126704887070720l);
  EXPECT_EQ(1375040258, Unix64Time(time).get_offset());

  auto const time_zone = get_time_zone("US/Eastern");
  TimeParts const parts = get_parts(time, *time_zone);
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
  EXPECT_THROW(get_parts(Time::INVALID, "US/Eastern"), InvalidTimeError);
}

TEST(Time, get_parts_display) {
  auto const time = from_utc(2016/MAY/28, Daytime(16, 30, 0));

  set_display_time_zone("US/Eastern");  // EDT = UTC-04:00
  auto parts = get_parts(time, DTZ);
  EXPECT_EQ(2016,       parts.date.year);
  EXPECT_EQ(MAY,        parts.date.month);
  EXPECT_EQ(27,         parts.date.day);
  EXPECT_EQ(12,         parts.daytime.hour);
  EXPECT_EQ(30,         parts.daytime.minute);

  set_display_time_zone("UTC");
  parts = get_parts(time, DTZ);
  EXPECT_EQ(2016,       parts.date.year);
  EXPECT_EQ(MAY,        parts.date.month);
  EXPECT_EQ(27,         parts.date.day);
  EXPECT_EQ(16,         parts.daytime.hour);
  EXPECT_EQ(30,         parts.daytime.minute);

  set_display_time_zone("Asia/Tokyo");  // JST = UTC+09:00
  parts = get_parts(time, DTZ);
  EXPECT_EQ(2016,       parts.date.year);
  EXPECT_EQ(MAY,        parts.date.month);
  EXPECT_EQ(28,         parts.date.day);
  EXPECT_EQ(1,          parts.daytime.hour);
  EXPECT_EQ(30,         parts.daytime.minute);

  set_display_time_zone("Asia/Kolkata");  // IST = UTC+05:30
  parts = get_parts(time, DTZ);
  EXPECT_EQ(2016,       parts.date.year);
  EXPECT_EQ(MAY,        parts.date.month);
  EXPECT_EQ(27,         parts.date.day);
  EXPECT_EQ(22,         parts.daytime.hour);
  EXPECT_EQ(0 ,         parts.daytime.minute);
}

TEST(Time, default_format) {
  EXPECT_EQ("%Y-%m-%d %H:%M:%S %~Z", TimeFormat::get_default().get_pattern());
}

TEST(Time, ostream) {
  auto const tz = get_time_zone("US/Eastern");
  auto const time = from_local(2013/JUL/29, Daytime(18, 27, 13.6316313), *tz);
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
  auto const time = from_local(2013/JUL/29, Daytime(18, 27, 13.6316313), *tz);

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

  auto const date = get_utc_date<Date>(time);
  auto const ymd = get_ymd(date);
  EXPECT_EQ(2013, ymd.year);
  EXPECT_EQ(6, ymd.month);
  EXPECT_EQ(21, ymd.day);

  auto const daytime = get_utc_daytime<Daytime>(time);
  auto const hms = get_hms(daytime);
  EXPECT_EQ(19, hms.hour);
  EXPECT_EQ(43, hms.minute);
  EXPECT_EQ(52.0, hms.second);
}

