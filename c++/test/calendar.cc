#pragma GCC diagnostic ignored "-Wparentheses"

#include "aslib/filename.hh"
#include "cron/calendar.hh"
#include "cron/ez.hh"
#include "gtest/gtest.h"

using namespace aslib;
using namespace cron;
using namespace cron::ez;

//------------------------------------------------------------------------------
// Class AllCalendar.

TEST(AllCalendar, contains0) {
  AllCalendar const cal;
  EXPECT_TRUE (cal.contains(1970/JAN/ 1));
  EXPECT_TRUE (cal.contains(1973/DEC/ 3));
  EXPECT_TRUE (cal.contains(2013/JUL/11));
  EXPECT_TRUE (cal.contains(2013/JUL/12));
  EXPECT_TRUE (cal.contains(2013/JUL/13));
  EXPECT_TRUE (cal.contains(2013/JUL/14));
  EXPECT_TRUE (cal.contains(2013/JUL/15));
  EXPECT_TRUE (cal[2013/JUL/14]);
  EXPECT_TRUE (cal[2013/JUL/15]);
  EXPECT_TRUE (cal[Date::MIN]);
  EXPECT_TRUE (cal[Date::MAX]);
  EXPECT_FALSE(cal.contains(Date::INVALID));
  EXPECT_FALSE(cal.contains(Date::MISSING));
}

TEST(AllCalendar, contains1) {
  AllCalendar const cal;
  EXPECT_TRUE (cal.contains(Date16::from_ymd(1970,  0, 0)));
  EXPECT_TRUE (cal.contains(Date16::from_ymd(1973, 11, 2)));
  EXPECT_TRUE (cal.contains(Date16::from_ymd(2013,  6, 10)));
  EXPECT_TRUE (cal.contains(Date16::from_ymd(2013,  6, 11)));
  EXPECT_TRUE (cal.contains(Date16::from_ymd(2013,  6, 12)));
  EXPECT_TRUE (cal.contains(Date16::from_ymd(2013,  6, 13)));
  EXPECT_TRUE (cal.contains(Date16::from_ymd(2013,  6, 14)));
  EXPECT_TRUE (cal[Date16::from_ymd(2013,  6, 13)]);
  EXPECT_TRUE (cal[Date16::from_ymd(2013,  6, 14)]);
  EXPECT_TRUE (cal[Date16::MIN]);
  EXPECT_TRUE (cal[Date16::MAX]);
  EXPECT_FALSE(cal.contains(Date16::INVALID));
  EXPECT_FALSE(cal.contains(Date16::MISSING));
}

//------------------------------------------------------------------------------
// Class WeekdaysCalendar.

TEST(WeekdaysCalendar, contains) {
  // Monday through Friday.
  WeekdaysCalendar const cal0({MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY});
  EXPECT_TRUE (cal0.contains(1970/JAN/ 1));
  EXPECT_TRUE (cal0.contains(1973/DEC/ 3));
  EXPECT_TRUE (cal0.contains(2013/JUL/11));
  EXPECT_TRUE (cal0.contains(2013/JUL/12));
  EXPECT_FALSE(cal0.contains(2013/JUL/13));
  EXPECT_FALSE(cal0.contains(2013/JUL/14));
  EXPECT_TRUE (cal0.contains(2013/JUL/15));
  EXPECT_FALSE(cal0[2013/JUL/14]);
  EXPECT_TRUE (cal0[2013/JUL/15]);

  // Thursdays and Sundays.
  WeekdaysCalendar const cal1({SUNDAY, THURSDAY});
  EXPECT_TRUE (cal1.contains(1970/JAN/ 1));
  EXPECT_FALSE(cal1.contains(1973/DEC/ 3));
  EXPECT_TRUE (cal1.contains(2013/JUL/11));
  EXPECT_FALSE(cal1.contains(2013/JUL/12));
  EXPECT_FALSE(cal1.contains(2013/JUL/13));
  EXPECT_TRUE (cal1.contains(2013/JUL/14));
  EXPECT_FALSE(cal1.contains(2013/JUL/15));
  EXPECT_TRUE (cal1[2013/JUL/14]);
  EXPECT_FALSE(cal1[2013/JUL/15]);
}

TEST(WeekdaysCalendar, shift) {
  // Monday through Friday.
  WeekdaysCalendar const cal({MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY});
  auto const date = 2013/JUL/11;

  EXPECT_EQ(2013/JUL/11, date + cal.DAY *   0);

  EXPECT_EQ(2013/JUL/12, date + cal.DAY *   1);
  EXPECT_EQ(2013/JUL/15, date + cal.DAY *   2);
  EXPECT_EQ(2013/JUL/16, date + cal.DAY *   3);
  EXPECT_EQ(2013/JUL/18, date + cal.DAY *   5);
  EXPECT_EQ(2013/JUL/25, date + cal.DAY *  10);
  EXPECT_EQ(2013/AUG/ 1, date + cal.DAY *  15);
  EXPECT_EQ(2013/AUG/15, date + cal.DAY *  25);
  EXPECT_EQ(2014/JUL/10, date + cal.DAY * 260);
  EXPECT_EQ(2014/JUL/11, date + cal.DAY * 261);
  EXPECT_EQ(2014/JUL/11, date + 261 * cal.DAY);

  EXPECT_EQ(2013/JUL/10, date + cal.DAY *  -1);
  EXPECT_EQ(2013/JUL/ 9, date + cal.DAY *  -2);
  EXPECT_EQ(2013/JUL/ 5, date + cal.DAY *  -4);
  EXPECT_EQ(2013/JUN/28, date + cal.DAY *  -9);
  EXPECT_EQ(2013/JUN/28, date - 9 * cal.DAY);

  EXPECT_THROW((9900/JAN/ 1 + 600000 * cal.DAY).is_invalid(), DateRangeError);
  EXPECT_THROW((1600/DEC/ 1 - 600000 * cal.DAY).is_invalid(), DateRangeError);
}

TEST(WeekdaysCalendar, nearest) {
  // Monday through Friday.
  WeekdaysCalendar const cal({MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY});

  EXPECT_EQ(2013/JUL/23, cal.nearest(2013/JUL/23, false));
  EXPECT_EQ(2013/JUL/23, cal.nearest(2013/JUL/23, true));
  EXPECT_EQ(2013/JUL/23, 2013/JUL/23 << cal);
  EXPECT_EQ(2013/JUL/23, 2013/JUL/23 >> cal);

  EXPECT_EQ(2013/JUL/26, cal.nearest(2013/JUL/28, false));
  EXPECT_EQ(2013/JUL/29, cal.nearest(2013/JUL/28, true));
  EXPECT_EQ(2013/JUL/26, 2013/JUL/28 << cal);
  EXPECT_EQ(2013/JUL/29, 2013/JUL/28 >> cal);

  auto date = 2013/JUL/28;
  date <<= cal;
  EXPECT_EQ(2013/JUL/26, date);
}

//------------------------------------------------------------------------------
// Class HolidayCalendar.

TEST(HolidayCalendar, load) {
  HolidayCalendar cal = load_holiday_calendar(fs::Filename("holidays.cal"));
  EXPECT_EQ(cal.get_min(), 2010/JAN/ 1);
  EXPECT_EQ(cal.get_max(), 2021/JAN/ 1);
  EXPECT_FALSE(cal[2012/JUL/ 3]);
  EXPECT_TRUE (cal[2012/JUL/ 4]);
  EXPECT_FALSE(cal[2012/JUL/ 5]);
}

//------------------------------------------------------------------------------
// Class WorkdayCalendar.

TEST(WorkdayCalendar, contains) {
  WorkdayCalendar cal(
    {MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY},
    fs::Filename("holidays.cal"));
  EXPECT_FALSE(cal[2012/JUL/ 1]);  // Sunday
  EXPECT_TRUE (cal[2012/JUL/ 2]);
  EXPECT_TRUE (cal[2012/JUL/ 3]);
  EXPECT_FALSE(cal[2012/JUL/ 4]);  // holiday
  EXPECT_TRUE (cal[2012/JUL/ 5]);
  EXPECT_TRUE (cal[2012/JUL/ 6]);
  EXPECT_FALSE(cal[2012/JUL/ 7]);  // Saturday
}

