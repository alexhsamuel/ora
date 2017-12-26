#include "ora.hh"
#include "gtest/gtest.h"

using namespace aslib;
using namespace ora;
using namespace ora::ez;

//------------------------------------------------------------------------------
// Class Time
//------------------------------------------------------------------------------

TEST(Time, comparisons) {
  Time const t = from_local_parts(2013, 6, 29, 18, 27, 13, "US/Eastern");
  Time const i = Time::INVALID;
  Time const m = Time::MISSING;

  EXPECT_FALSE(t <  t);
  EXPECT_FALSE(t <  i);
  EXPECT_FALSE(t <  m);
  EXPECT_TRUE (i <  t);
  EXPECT_FALSE(i <  i);
  EXPECT_TRUE (i <  m);
  EXPECT_TRUE (m <  t);
  EXPECT_FALSE(m <  i);
  EXPECT_FALSE(m <  m);

  EXPECT_TRUE (t == t);
  EXPECT_FALSE(t == i);
  EXPECT_FALSE(t == m);
  EXPECT_FALSE(i == t);
  EXPECT_TRUE (i == i);
  EXPECT_FALSE(i == m);
  EXPECT_FALSE(m == t);
  EXPECT_FALSE(m == i);
  EXPECT_TRUE (m == m);

  EXPECT_TRUE (t >= t);
  EXPECT_TRUE (t >= i);
  EXPECT_TRUE (t >= m);
  EXPECT_FALSE(i >= t);
  EXPECT_TRUE (i >= i);
  EXPECT_FALSE(i >= m);
  EXPECT_FALSE(m >= t);
  EXPECT_TRUE (m >= i);
  EXPECT_TRUE (m >= m);

  EXPECT_TRUE (t == from_local_parts(2013, 6, 29, 18, 27, 13,   "US/Eastern"));
  EXPECT_TRUE (t >  from_local_parts(2013, 6, 28, 18, 27, 13,   "US/Eastern"));
  EXPECT_TRUE (t <  from_local_parts(2013, 6, 29, 18, 27, 13.1, "US/Eastern"));
  EXPECT_FALSE(t >  from_local_parts(2013, 6, 29, 18, 27, 13.1, "US/Eastern"));
}

TEST(Time, from_parts) {
  auto const daytime = from_hms(18, 27, 13);
  auto const tz = get_time_zone("US/Eastern");
  auto const time0 = from_local<Unix32Time>(2013/JUL/29, daytime, *tz);
  EXPECT_EQ(1375136833, time0.get_offset());

  Time::Offset const offset = 4262126704878682112l;
  Time const time1 = time::from_offset(offset);
  EXPECT_EQ(offset, time1.get_offset());
  TimeParts const parts1 = get_parts(time1, "US/Eastern");
  EXPECT_EQ(2013, parts1.date.year);
  EXPECT_EQ(7, parts1.date.month);
  EXPECT_EQ(28, parts1.date.day);

  auto const time2 = from_local(2013/JUL/28, from_hms(15, 37, 38), *tz);
  EXPECT_EQ(offset, time2.get_offset());

  auto const time3 = from_local_parts(2013, 7, 28, 15, 37, 38, *tz);
  EXPECT_EQ(offset, time3.get_offset());
}

TEST(Time, from_parts_dst) {
  auto const tz = get_time_zone("US/Eastern");

  // Test transition to DST.
  Date const dst0 = 2013/MAR/10;
  EXPECT_EQ(from_local(dst0, from_hms(6, 59, 0), *UTC), from_local(dst0, from_hms(1, 59, 0), *tz));
  EXPECT_EQ(from_local(dst0, from_hms(7,  0, 0), *UTC), from_local(dst0, from_hms(3,  0, 0), *tz));
  EXPECT_EQ(from_local(dst0, from_hms(7,  0, 0), *UTC), from_local(dst0, from_hms(3,  0, 0), *tz, false));

  // Test transition from DST.
  Date const dst1 = 2013/NOV/3;
  EXPECT_EQ(from_local(dst1, from_hms(4, 59, 0), *UTC), from_local(dst1, from_hms(0, 59, 0), *tz));
  EXPECT_EQ(from_local(dst1, from_hms(5,  0, 0), *UTC), from_local(dst1, from_hms(1,  0, 0), *tz));
  EXPECT_EQ(from_local(dst1, from_hms(5,  0, 0), *UTC), from_local(dst1, from_hms(1,  0, 0), *tz, true));
  EXPECT_EQ(from_local(dst1, from_hms(5, 59, 0), *UTC), from_local(dst1, from_hms(1, 59, 0), *tz));
  EXPECT_EQ(from_local(dst1, from_hms(5, 59, 0), *UTC), from_local(dst1, from_hms(1, 59, 0), *tz, true));
  EXPECT_EQ(from_local(dst1, from_hms(6,  0, 0), *UTC), from_local(dst1, from_hms(1,  0, 0), *tz, false));
  EXPECT_EQ(from_local(dst1, from_hms(6, 59, 0), *UTC), from_local(dst1, from_hms(1, 59, 0), *tz, false));
  EXPECT_EQ(from_local(dst1, from_hms(7,  0, 0), *UTC), from_local(dst1, from_hms(2,  0, 0), *tz));
  EXPECT_EQ(from_local(dst1, from_hms(7,  0, 0), *UTC), from_local(dst1, from_hms(2,  0, 0), *tz, false));
}

TEST(Time, from_parts_invalid) {
  auto const tz = get_time_zone("US/Eastern");

  EXPECT_THROW(from_local(Date::INVALID, from_hms(0, 0, 0), *tz), InvalidDateError);
  EXPECT_THROW(from_local(Date::MISSING, from_hms(0, 0, 0), *tz), InvalidDateError);

  EXPECT_THROW(from_local(2013/JUL/28, Daytime::INVALID , *tz), InvalidDaytimeError);
  EXPECT_THROW(from_local(2013/JUL/28, from_hms(24, 0, 0), *tz), InvalidDaytimeError);
  EXPECT_THROW(from_local(2013/JUL/28, from_hms(0, 60, 0), *tz), InvalidDaytimeError);
  EXPECT_THROW(from_local(2013/JUL/28, from_hms(0, 0, 60), *tz), InvalidDaytimeError);

  EXPECT_THROW(from_local_parts(10000,  1,  1,  0,  0,  0, *tz), InvalidDateError);
  EXPECT_THROW(from_local_parts( 2013, 13,  1,  0,  0,  0, *tz), InvalidDateError);
  EXPECT_THROW(from_local_parts( 2013,  1, 32,  0,  0,  0, *tz), InvalidDateError);
  EXPECT_THROW(from_local_parts( 2013,  1,  1, 24,  0,  0, *tz), InvalidDaytimeError);
  EXPECT_THROW(from_local_parts( 2013,  1,  1,  0, 60,  0, *tz), InvalidDaytimeError);
  EXPECT_THROW(from_local_parts( 2013,  1,  1,  0,  0, 60, *tz), InvalidDaytimeError);

  EXPECT_TRUE (from_local_parts( 2013,  3, 10,  1, 59, 59, *tz).is_valid());
  EXPECT_THROW(from_local_parts( 2013,  3, 10,  2,  0,  0, *tz), NonexistentDateDaytime);
  EXPECT_THROW(from_local_parts( 2013,  3, 10,  2, 59, 59, *tz), NonexistentDateDaytime);
  EXPECT_TRUE (from_local_parts( 2013,  3, 10,  3,  0,  0, *tz).is_valid());
  EXPECT_TRUE (from_local_parts( 2013,  3, 10,  3,  0,  0, *tz, false).is_valid());
}

TEST(Time, get_parts) {
  // 2013 July 28 15:37:38.125 EDT [UTC-4].
  auto const time = time::from_offset(4262126704887070720l);
  EXPECT_EQ(1375040258, Unix64Time(time).get_offset());

  auto const time_zone = get_time_zone("US/Eastern");
  TimeParts const parts = get_parts(time, *time_zone);
  EXPECT_EQ(2013,       parts.date.year);
  EXPECT_EQ(7,          parts.date.month);
  EXPECT_EQ(28,         parts.date.day);
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
  auto const time = from_utc(2016/MAY/28, from_hms(16, 30, 0));

  set_display_time_zone("US/Eastern");  // EDT = UTC-04:00
  auto parts = get_parts(time, DTZ);
  EXPECT_EQ(2016,       parts.date.year);
  EXPECT_EQ(MAY,        parts.date.month);
  EXPECT_EQ(28,         parts.date.day);
  EXPECT_EQ(12,         parts.daytime.hour);
  EXPECT_EQ(30,         parts.daytime.minute);

  set_display_time_zone("UTC");
  parts = get_parts(time, DTZ);
  EXPECT_EQ(2016,       parts.date.year);
  EXPECT_EQ(MAY,        parts.date.month);
  EXPECT_EQ(28,         parts.date.day);
  EXPECT_EQ(16,         parts.daytime.hour);
  EXPECT_EQ(30,         parts.daytime.minute);

  set_display_time_zone("Asia/Tokyo");  // JST = UTC+09:00
  parts = get_parts(time, DTZ);
  EXPECT_EQ(2016,       parts.date.year);
  EXPECT_EQ(MAY,        parts.date.month);
  EXPECT_EQ(29,         parts.date.day);
  EXPECT_EQ(1,          parts.daytime.hour);
  EXPECT_EQ(30,         parts.daytime.minute);

  set_display_time_zone("Asia/Kolkata");  // IST = UTC+05:30
  parts = get_parts(time, DTZ);
  EXPECT_EQ(2016,       parts.date.year);
  EXPECT_EQ(MAY,        parts.date.month);
  EXPECT_EQ(28,         parts.date.day);
  EXPECT_EQ(22,         parts.daytime.hour);
  EXPECT_EQ(0 ,         parts.daytime.minute);
}

//------------------------------------------------------------------------------
// Class Unix32Time.

TEST(Unix32Time, zero) {
  auto const time = time::from_offset<Unix32Time>(1374522232);

  auto const date = get_utc_date<Date>(time);
  auto const ymd = get_ymd(date);
  EXPECT_EQ(2013, ymd.year);
  EXPECT_EQ(7, ymd.month);
  EXPECT_EQ(22, ymd.day);

  auto const daytime = get_utc_daytime<Daytime>(time);
  auto const hms = get_hms(daytime);
  EXPECT_EQ(19, hms.hour);
  EXPECT_EQ(43, hms.minute);
  EXPECT_EQ(52.0, hms.second);
}

