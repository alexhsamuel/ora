#pragma GCC diagnostic ignored "-Wparentheses"

#include "ora/lib/filename.hh"
#include "ora.hh"
#include "ora/calendar.hh"
#include "gtest/gtest.h"

using namespace ora::lib;
using namespace ora;
using namespace ora::ez;

//------------------------------------------------------------------------------
// Class AllCalendar.

TEST(AllCalendar, contains0) {
  auto const cal = make_const_calendar({Date::MIN, Date::MAX}, true);
  EXPECT_TRUE (cal.contains(1970/JAN/ 1));
  EXPECT_TRUE (cal.contains(1973/DEC/ 3));
  EXPECT_TRUE (cal.contains(2013/JUL/11));
  EXPECT_TRUE (cal.contains(2013/JUL/12));
  EXPECT_TRUE (cal.contains(2013/JUL/13));
  EXPECT_TRUE (cal.contains(2013/JUL/14));
  EXPECT_TRUE (cal.contains(2013/JUL/15));
  EXPECT_TRUE (cal.contains(2013/JUL/14));
  EXPECT_TRUE (cal.contains(2013/JUL/15));
  EXPECT_TRUE (cal.contains(Date::MIN));
  EXPECT_TRUE (cal.contains(Date::MAX));
  EXPECT_THROW(cal.contains(Date::INVALID), CalendarRangeError);
  EXPECT_THROW(cal.contains(Date::MISSING), CalendarRangeError);
}

TEST(AllCalendar, contains1) {
  auto const cal = make_const_calendar({Date::MIN, Date::MAX}, true);
  EXPECT_TRUE (cal.contains(from_ymd<Date16>(1970,  1,  1)));
  EXPECT_TRUE (cal.contains(from_ymd<Date16>(1973, 12,  3)));
  EXPECT_TRUE (cal.contains(from_ymd<Date16>(2013,  7, 11)));
  EXPECT_TRUE (cal.contains(from_ymd<Date16>(2013,  7, 12)));
  EXPECT_TRUE (cal.contains(from_ymd<Date16>(2013,  7, 13)));
  EXPECT_TRUE (cal.contains(from_ymd<Date16>(2013,  7, 14)));
  EXPECT_TRUE (cal.contains(from_ymd<Date16>(2013,  7, 15)));
  EXPECT_TRUE (cal.contains(from_ymd<Date16>(2013,  7, 14)));
  EXPECT_TRUE (cal.contains(from_ymd<Date16>(2013,  7, 15)));
  EXPECT_TRUE (cal.contains(Date16::MIN));
  EXPECT_TRUE (cal.contains(Date16::MAX));
  EXPECT_THROW(cal.contains(Date16::INVALID), CalendarRangeError);
  EXPECT_THROW(cal.contains(Date16::MISSING), CalendarRangeError);
}

//------------------------------------------------------------------------------
// Class WeekdaysCalendar.

TEST(WeekdaysCalendar, contains) {
  // Monday through Friday.
  auto const cal0 = make_weekday_calendar(
    {Date::MIN, Date::MAX}, 
    (bool const[]){true, true, true, true, true, false, false});
  EXPECT_TRUE (cal0.contains(1970/JAN/ 1));
  EXPECT_TRUE (cal0.contains(1973/DEC/ 3));
  EXPECT_TRUE (cal0.contains(2013/JUL/11));
  EXPECT_TRUE (cal0.contains(2013/JUL/12));
  EXPECT_FALSE(cal0.contains(2013/JUL/13));
  EXPECT_FALSE(cal0.contains(2013/JUL/14));
  EXPECT_TRUE (cal0.contains(2013/JUL/15));
  EXPECT_FALSE(cal0.contains(2013/JUL/14));
  EXPECT_TRUE (cal0.contains(2013/JUL/15));

  // Thursdays and Sundays.
  auto const cal1 = make_weekday_calendar(
    {Date::MIN, Date::MAX}, 
    (bool const[]){false, false, false, true, false, false, true});
  EXPECT_TRUE (cal1.contains(1970/JAN/ 1));
  EXPECT_FALSE(cal1.contains(1973/DEC/ 3));
  EXPECT_TRUE (cal1.contains(2013/JUL/11));
  EXPECT_FALSE(cal1.contains(2013/JUL/12));
  EXPECT_FALSE(cal1.contains(2013/JUL/13));
  EXPECT_TRUE (cal1.contains(2013/JUL/14));
  EXPECT_FALSE(cal1.contains(2013/JUL/15));
  EXPECT_TRUE (cal1.contains(2013/JUL/14));
  EXPECT_FALSE(cal1.contains(2013/JUL/15));
}

TEST(WeekdaysCalendar, shift) {
  // Monday through Friday.
  auto const cal = make_weekday_calendar(
    {Date::MIN, Date::MAX}, 
    (bool const[]){true, true, true, true, true, false, false});
  auto const date = 2013/JUL/11;

  auto const day = cal.DAY();
  EXPECT_EQ(2013/JUL/11, date + day *   0);

  EXPECT_EQ(2013/JUL/12, date + day *   1);
  EXPECT_EQ(2013/JUL/15, date + day *   2);
  EXPECT_EQ(2013/JUL/16, date + day *   3);
  EXPECT_EQ(2013/JUL/18, date + day *   5);
  EXPECT_EQ(2013/JUL/25, date + day *  10);
  EXPECT_EQ(2013/AUG/ 1, date + day *  15);
  EXPECT_EQ(2013/AUG/15, date + day *  25);
  EXPECT_EQ(2014/JUL/10, date + day * 260);
  EXPECT_EQ(2014/JUL/11, date + day * 261);
  EXPECT_EQ(2014/JUL/11, date + 261 * day);

  EXPECT_EQ(2013/JUL/10, date + day *  -1);
  EXPECT_EQ(2013/JUL/ 9, date + day *  -2);
  EXPECT_EQ(2013/JUL/ 5, date + day *  -4);
  EXPECT_EQ(2013/JUN/28, date + day *  -9);
  EXPECT_EQ(2013/JUN/28, date - 9 * day);

  EXPECT_THROW((9900/JAN/ 1 + 600000 * day).is_invalid(), DateRangeError);
  EXPECT_THROW((1600/DEC/ 1 - 600000 * day).is_invalid(), DateRangeError);
}

TEST(WeekdaysCalendar, nearest) {
  // Monday through Friday.
  auto const cal = make_weekday_calendar(
    {Date::MIN, Date::MAX}, 
    (bool const[]){true, true, true, true, true, false, false});

  EXPECT_EQ(2013/JUL/23, cal.before(2013/JUL/23));
  EXPECT_EQ(2013/JUL/23, cal.after(2013/JUL/23));
  EXPECT_EQ(2013/JUL/22, 2013/JUL/23 << cal);
  EXPECT_EQ(2013/JUL/24, 2013/JUL/23 >> cal);

  EXPECT_EQ(2013/JUL/26, cal.before(2013/JUL/28));
  EXPECT_EQ(2013/JUL/29, cal.after(2013/JUL/28));
  EXPECT_EQ(2013/JUL/26, 2013/JUL/28 << cal);
  EXPECT_EQ(2013/JUL/29, 2013/JUL/28 >> cal);

  auto date = 2013/JUL/28;
  date <<= cal;
  EXPECT_EQ(2013/JUL/26, date);
}

//------------------------------------------------------------------------------
// Calendar files

TEST(Calendar, load) {
  auto const cal = load_calendar(fs::Filename("holidays.cal"));
  EXPECT_EQ(cal.range().min, 2010/JAN/ 1);
  EXPECT_EQ(cal.range().max, 2021/JAN/ 1);
  EXPECT_FALSE(cal.contains(2012/JUL/ 3));
  EXPECT_TRUE (cal.contains(2012/JUL/ 4));
  EXPECT_FALSE(cal.contains(2012/JUL/ 5));
}

