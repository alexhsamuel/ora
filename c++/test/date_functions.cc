#include "cron/date.hh"
#include "cron/ez.hh"
#include "cron/format.hh"
#include "gtest/gtest.h"

using namespace cron;
using namespace cron::date;
using namespace cron::ez;

//------------------------------------------------------------------------------

inline bool
check(
  OrdinalDate const& ordinal_date,
  Year const year,
  Ordinal const ordinal)
{
  return ordinal_date.year == year && ordinal_date.ordinal == ordinal;
}


inline bool
check(
  WeekDate const& week_date,
  Year const week_year,
  Week const week,
  Weekday const weekday)
{
  return 
       week_date.week_year == week_year
    && week_date.week == week
    && week_date.weekday == weekday;
}


inline bool
check(
  YmdDate const& ymd,
  Year const year,
  Month const month,
  Day const day)
{
  return ymd.year == year && ymd.month == month && ymd.day == day;
}


TEST(get_ordinal_date, Date) {
  EXPECT_TRUE(check(get_ordinal_date(   1/JAN/ 1),    1,   0));
  EXPECT_TRUE(check(get_ordinal_date(   1/DEC/31),    1, 364));

  EXPECT_TRUE(check(get_ordinal_date(   2/JAN/ 1),    2,   0));
  EXPECT_TRUE(check(get_ordinal_date(   2/DEC/31),    2, 364));

  EXPECT_TRUE(check(get_ordinal_date(   4/JAN/ 1),    4,   0));
  EXPECT_TRUE(check(get_ordinal_date(   4/FEB/28),    4,  58));
  EXPECT_TRUE(check(get_ordinal_date(   4/FEB/29),    4,  59));
  EXPECT_TRUE(check(get_ordinal_date(   4/MAR/ 1),    4,  60));
  EXPECT_TRUE(check(get_ordinal_date(   4/DEC/31),    4, 365));

  EXPECT_TRUE(check(get_ordinal_date(9999/JAN/ 1), 9999,   0));
  EXPECT_TRUE(check(get_ordinal_date(9999/DEC/31), 9999, 364));
}

TEST(get_ordinal_date, Date16) {
  EXPECT_TRUE(check(get_ordinal_date(Date16(2002/JAN/ 1)), 2002,   0));
  EXPECT_TRUE(check(get_ordinal_date(Date16(2002/DEC/31)), 2002, 364));

  EXPECT_TRUE(check(get_ordinal_date(Date16(2004/JAN/ 1)), 2004,   0));
  EXPECT_TRUE(check(get_ordinal_date(Date16(2004/FEB/28)), 2004,  58));
  EXPECT_TRUE(check(get_ordinal_date(Date16(2004/FEB/29)), 2004,  59));
  EXPECT_TRUE(check(get_ordinal_date(Date16(2004/MAR/ 1)), 2004,  60));
  EXPECT_TRUE(check(get_ordinal_date(Date16(2004/DEC/31)), 2004, 365));
}

TEST(get_ordinal_date, invalid) {
  EXPECT_THROW(get_ordinal_date(Date::INVALID), InvalidDateError);
  EXPECT_THROW(get_ordinal_date(Date::MISSING), InvalidDateError);
  EXPECT_THROW(get_ordinal_date(Date16::INVALID), InvalidDateError);
  EXPECT_THROW(get_ordinal_date(Date16::MISSING), InvalidDateError);
}

TEST(get_weekday, Date) {
  EXPECT_EQ(get_weekday(   1/JAN/ 1), MONDAY);
  EXPECT_EQ(get_weekday(   1/JAN/ 2), TUESDAY);
  EXPECT_EQ(get_weekday(   1/JAN/ 3), WEDNESDAY);
  EXPECT_EQ(get_weekday(   1/JAN/ 4), THURSDAY);
  EXPECT_EQ(get_weekday(   1/JAN/ 5), FRIDAY);
  EXPECT_EQ(get_weekday(   1/JAN/ 6), SATURDAY);
  EXPECT_EQ(get_weekday(   1/JAN/ 7), SUNDAY);
  EXPECT_EQ(get_weekday(   1/JAN/ 8), MONDAY);
}

TEST(get_weekday, invalid) {
  EXPECT_THROW(get_weekday(Date::INVALID), InvalidDateError);
  EXPECT_THROW(get_weekday(Date::MISSING), InvalidDateError);
  EXPECT_THROW(get_weekday(Date16::INVALID), InvalidDateError);
  EXPECT_THROW(get_weekday(Date16::MISSING), InvalidDateError);
}

TEST(get_week_date, Date) {
  EXPECT_TRUE(check(get_week_date(   1/JAN/ 1),    1,  0, 0));
  EXPECT_TRUE(check(get_week_date(   1/JAN/ 2),    1,  0, 1));

  EXPECT_TRUE(check(get_week_date(2005/JAN/ 1), 2004, 52, 5));
  EXPECT_TRUE(check(get_week_date(2005/JAN/ 2), 2004, 52, 6));
  EXPECT_TRUE(check(get_week_date(2005/DEC/31), 2005, 51, 5));
  EXPECT_TRUE(check(get_week_date(2007/JAN/ 1), 2007,  0, 0));
  EXPECT_TRUE(check(get_week_date(2007/DEC/30), 2007, 51, 6));
  EXPECT_TRUE(check(get_week_date(2007/DEC/31), 2008,  0, 0));
  EXPECT_TRUE(check(get_week_date(2008/JAN/ 1), 2008,  0, 1));
  EXPECT_TRUE(check(get_week_date(2008/DEC/28), 2008, 51, 6));
  EXPECT_TRUE(check(get_week_date(2008/DEC/29), 2009,  0, 0));
  EXPECT_TRUE(check(get_week_date(2008/DEC/30), 2009,  0, 1));
  EXPECT_TRUE(check(get_week_date(2008/DEC/31), 2009,  0, 2));
  EXPECT_TRUE(check(get_week_date(2009/JAN/ 1), 2009,  0, 3));
  EXPECT_TRUE(check(get_week_date(2009/DEC/31), 2009, 52, 3));
  EXPECT_TRUE(check(get_week_date(2010/JAN/ 1), 2009, 52, 4));
  EXPECT_TRUE(check(get_week_date(2010/JAN/ 2), 2009, 52, 5));
  EXPECT_TRUE(check(get_week_date(2010/JAN/ 3), 2009, 52, 6));
}

TEST(get_week_date, Date16) {
  {
    auto const wd = get_week_date(Date16(2008/DEC/28));
    EXPECT_EQ(2008, wd.week_year);
    EXPECT_EQ(  51, wd.week);
    EXPECT_EQ(   6, wd.weekday);
  }
  {
    auto const wd = get_week_date(Date16(2008/DEC/29));
    EXPECT_EQ(2009, wd.week_year);
    EXPECT_EQ(   0, wd.week);
    EXPECT_EQ(   0, wd.weekday);
  }
}

TEST(get_week_date, invalid) {
  EXPECT_THROW(get_week_date(Date::INVALID), InvalidDateError);
  EXPECT_THROW(get_week_date(Date::MISSING), InvalidDateError);
  EXPECT_THROW(get_week_date(Date16::INVALID), InvalidDateError);
  EXPECT_THROW(get_week_date(Date16::MISSING), InvalidDateError);
}

TEST(get_ymd, thorough_Date) {
  for (Year y = 1; y <= 9999; y += 13)
    for (Month m = 0; m < 12; ++m)
      for (Day d = 0; d < 28; d += 8) {
        auto const ymd = get_ymd(Date(y, m, d));
        EXPECT_EQ(y, ymd.year);
        EXPECT_EQ(m, ymd.month);
        EXPECT_EQ(d, ymd.day);
      }
}

TEST(get_ymd, thorough_Date16) {
  for (Year y = 1970; y <= 2149; y += 13)
    for (Month m = 0; m < 12; ++m)
      for (Day d = 0; d < 28; d += 8) {
        auto const ymd = get_ymd(Date16(y, m, d));
        EXPECT_EQ(y, ymd.year);
        EXPECT_EQ(m, ymd.month);
        EXPECT_EQ(d, ymd.day);
      }
}

TEST(get_ymd, edge) {
  EXPECT_TRUE(check(get_ymd(   1/JAN/ 1),    1,  0,  0));
  EXPECT_TRUE(check(get_ymd(   1/JAN/31),    1,  0, 30));
  EXPECT_TRUE(check(get_ymd(   1/FEB/ 1),    1,  1,  0));
  EXPECT_TRUE(check(get_ymd(   1/FEB/28),    1,  1, 27));
  EXPECT_TRUE(check(get_ymd(   1/MAR/ 1),    1,  2,  0));
  EXPECT_TRUE(check(get_ymd(   1/DEC/ 1),    1, 11,  0));
  EXPECT_TRUE(check(get_ymd(   1/DEC/31),    1, 11, 30));

  EXPECT_TRUE(check(get_ymd(2000/JAN/ 1), 2000,  0,  0));
  EXPECT_TRUE(check(get_ymd(2000/JAN/31), 2000,  0, 30));
  EXPECT_TRUE(check(get_ymd(2000/FEB/ 1), 2000,  1,  0));
  EXPECT_TRUE(check(get_ymd(2000/FEB/28), 2000,  1, 27));
  EXPECT_TRUE(check(get_ymd(2000/FEB/29), 2000,  1, 28));
  EXPECT_TRUE(check(get_ymd(2000/MAR/ 1), 2000,  2,  0));
  EXPECT_TRUE(check(get_ymd(2000/DEC/ 1), 2000, 11,  0));
  EXPECT_TRUE(check(get_ymd(2000/DEC/31), 2000, 11, 30));

  EXPECT_TRUE(check(get_ymd(9999/JAN/ 1), 9999,  0,  0));
  EXPECT_TRUE(check(get_ymd(9999/JAN/31), 9999,  0, 30));
  EXPECT_TRUE(check(get_ymd(9999/FEB/ 1), 9999,  1,  0));
  EXPECT_TRUE(check(get_ymd(9999/FEB/28), 9999,  1, 27));
  EXPECT_TRUE(check(get_ymd(9999/MAR/ 1), 9999,  2,  0));
  EXPECT_TRUE(check(get_ymd(9999/DEC/ 1), 9999, 11,  0));
  EXPECT_TRUE(check(get_ymd(9999/DEC/31), 9999, 11, 30));
}

TEST(get_ymd, invalid) {
  EXPECT_THROW(get_ymd(Date::INVALID), InvalidDateError);
  EXPECT_THROW(get_ymd(Date::MISSING), InvalidDateError);
  EXPECT_THROW(get_ymd(Date16::INVALID), InvalidDateError);
  EXPECT_THROW(get_ymd(Date16::MISSING), InvalidDateError);
}

TEST(get_ymdi, Date) {
  EXPECT_EQ(   10101, get_ymdi(   1/JAN/ 1));
  EXPECT_EQ(   10102, get_ymdi(   1/JAN/ 2));
  EXPECT_EQ(   10131, get_ymdi(   1/JAN/31));
  EXPECT_EQ(   10201, get_ymdi(   1/FEB/ 1));
  EXPECT_EQ(   10228, get_ymdi(   1/FEB/28));
  EXPECT_EQ(   10301, get_ymdi(   1/MAR/ 1));
  EXPECT_EQ(   11201, get_ymdi(   1/DEC/ 1));
  EXPECT_EQ(   11231, get_ymdi(   1/DEC/31));
  EXPECT_EQ(   20101, get_ymdi(   2/JAN/ 1));
  EXPECT_EQ(   20102, get_ymdi(   2/JAN/ 2));
  EXPECT_EQ(   21231, get_ymdi(   2/DEC/31));
  EXPECT_EQ(  100101, get_ymdi(  10/JAN/ 1));
  EXPECT_EQ(  991231, get_ymdi(  99/DEC/31));
  EXPECT_EQ( 1000101, get_ymdi( 100/JAN/ 1));
  EXPECT_EQ( 9991231, get_ymdi( 999/DEC/31));
  EXPECT_EQ(10000101, get_ymdi(1000/JAN/ 1));
  EXPECT_EQ(20000228, get_ymdi(2000/FEB/28));
  EXPECT_EQ(20000229, get_ymdi(2000/FEB/29));
  EXPECT_EQ(20000301, get_ymdi(2000/MAR/ 1));
  EXPECT_EQ(20010228, get_ymdi(2001/FEB/28));
  EXPECT_EQ(20010301, get_ymdi(2001/MAR/ 1));
  EXPECT_EQ(99990101, get_ymdi(9999/JAN/ 1));
  EXPECT_EQ(99991231, get_ymdi(9999/DEC/31));
}

TEST(get_ymdi, Date16) {
  EXPECT_EQ(20000228, get_ymdi(Date16(2000/FEB/28)));
  EXPECT_EQ(20000229, get_ymdi(Date16(2000/FEB/29)));
  EXPECT_EQ(20000301, get_ymdi(Date16(2000/MAR/ 1)));
  EXPECT_EQ(20010228, get_ymdi(Date16(2001/FEB/28)));
}

TEST(get_ymdi, invalid) {
  EXPECT_THROW(get_ymdi(Date::INVALID), InvalidDateError);
  EXPECT_THROW(get_ymdi(Date::MISSING), InvalidDateError);
  EXPECT_THROW(get_ymdi(Date16::INVALID), InvalidDateError);
  EXPECT_THROW(get_ymdi(Date16::MISSING), InvalidDateError);
}

TEST(days_after, Date) {
  EXPECT_EQ(   1/JAN/ 1, days_after(   1/JAN/ 1,       0));
  EXPECT_EQ(   1/JAN/ 2, days_after(   1/JAN/ 1,       1));
  EXPECT_EQ(   1/APR/11, days_after(   1/JAN/ 1,     100));
  EXPECT_EQ(   3/SEP/28, days_after(   1/JAN/ 1,    1000));
  EXPECT_EQ(  28/MAY/19, days_after(   1/JAN/ 1,   10000));
  EXPECT_EQ( 274/OCT/17, days_after(   1/JAN/ 1,  100000));
  EXPECT_EQ(2738/NOV/29, days_after(   1/JAN/ 1, 1000000));
  EXPECT_EQ(2738/NOV/30, days_after(   1/JAN/ 2, 1000000));
  EXPECT_EQ(2738/DEC/ 1, days_after(   1/JAN/ 3, 1000000));
  EXPECT_EQ(2000/JAN/ 1, days_after(1000/JAN/ 1,  365242));
  EXPECT_EQ(9999/DEC/31, days_after(   1/JAN/ 1, 3652058));
}

TEST(days_after, negative) {
  EXPECT_EQ(   1/JAN /1, days_after(   1/JAN/ 2,       -1));
  EXPECT_EQ(   1/JAN/ 1, days_after(   1/APR/11,     -100));
  EXPECT_EQ(   1/JAN/ 1, days_after(   3/SEP/28,    -1000));
  EXPECT_EQ(   1/JAN/ 1, days_after(  28/MAY/19,   -10000));
  EXPECT_EQ(   1/JAN/ 1, days_after( 274/OCT/17,  -100000));
  EXPECT_EQ(   1/JAN/ 1, days_after(2738/NOV/29, -1000000));
  EXPECT_EQ(   1/JAN/ 2, days_after(2738/NOV/30, -1000000));
  EXPECT_EQ(   1/JAN/ 3, days_after(2738/DEC/ 1, -1000000));
  EXPECT_EQ(1000/JAN/ 1, days_after(2000/JAN/ 1,  -365242));
  EXPECT_EQ(   1/JAN/ 1, days_after(9999/DEC/31, -3652058));
}

TEST(days_after, range) {
  EXPECT_THROW(days_after(   1/JAN/ 1,      -1), DateRangeError);
  EXPECT_THROW(days_after(   1/JAN/ 1, -100000), DateRangeError);
  EXPECT_THROW(days_after(   1/JAN/ 2,      -2), DateRangeError);
  EXPECT_THROW(days_after(   1/DEC/31,    -400), DateRangeError);
  EXPECT_THROW(days_after(   1/JAN/ 1, 3652059), DateRangeError);
  EXPECT_THROW(days_after(9999/JAN/ 1,     365), DateRangeError);
  EXPECT_THROW(days_after(9999/DEC/31,       1), DateRangeError);
  EXPECT_THROW(days_after(9999/DEC/31, 1000000), DateRangeError);
}

TEST(days_after, invalid) {
  EXPECT_THROW(days_after(Date::INVALID, 0), InvalidDateError);
  EXPECT_THROW(days_after(Date::MISSING, 0), InvalidDateError);
  EXPECT_THROW(days_after(Date16::INVALID, 0), InvalidDateError);
  EXPECT_THROW(days_after(Date16::MISSING, 0), InvalidDateError);

  EXPECT_THROW(days_after(Date::INVALID, 1), InvalidDateError);
  EXPECT_THROW(days_after(Date::MISSING, -1000), InvalidDateError);
  EXPECT_THROW(days_after(Date16::INVALID, 1000), InvalidDateError);
  EXPECT_THROW(days_after(Date16::MISSING, -1), InvalidDateError);
}

TEST(days_before, inverse) {
  for (int i = 0; i < 3652058; i += 137) {
    EXPECT_EQ(1/JAN/1, days_before(days_after(1/JAN/1, i), i));
    EXPECT_EQ(1/JAN/1, days_before(days_before(1/JAN/1, -i), i));

    EXPECT_EQ(9999/DEC/31, days_after(days_before(9999/DEC/31, i), i));
    EXPECT_EQ(9999/DEC/31, days_before(days_before(9999/DEC/31, i), -i));
  }
}

TEST(days_before, invalid) {
  EXPECT_THROW(days_before(Date::INVALID, 0), InvalidDateError);
  EXPECT_THROW(days_before(Date::MISSING, 0), InvalidDateError);
  EXPECT_THROW(days_before(Date16::INVALID, 0), InvalidDateError);
  EXPECT_THROW(days_before(Date16::MISSING, 0), InvalidDateError);

  EXPECT_THROW(days_before(Date::INVALID, 1), InvalidDateError);
  EXPECT_THROW(days_before(Date::MISSING, -1000), InvalidDateError);
  EXPECT_THROW(days_before(Date16::INVALID, 1000), InvalidDateError);
  EXPECT_THROW(days_before(Date16::MISSING, -1), InvalidDateError);
}

TEST(days_between, Date) {
  EXPECT_EQ(       0, days_between(   1/JAN/ 1,    1/JAN/ 1));
  EXPECT_EQ(       1, days_between(   1/JAN/ 1,    1/JAN/ 2));
  EXPECT_EQ(      -1, days_between(   1/JAN/ 2,    1/JAN/ 1));
  EXPECT_EQ( 3652058, days_between(   1/JAN/ 1, 9999/DEC/31));
  EXPECT_EQ(-3652058, days_between(9999/DEC/31,    1/JAN/ 1));
}

TEST(days_between, Date16) {
  EXPECT_EQ(    0, days_between(Date16(2000/JAN/ 1), Date16(2000/JAN/ 1)));
  EXPECT_EQ(  365, days_between(Date16(2000/JAN/ 1), Date16(2000/DEC/31)));
  EXPECT_EQ(  366, days_between(Date16(2000/JAN/ 1), Date16(2001/JAN/ 1)));
  EXPECT_EQ( 3653, days_between(Date16(2000/JAN/ 1), Date16(2010/JAN/ 1)));
  EXPECT_EQ(-3653, days_between(Date16(2010/JAN/ 1), Date16(2000/JAN/ 1)));
}

TEST(days_between, thorough) {
  for (int i = 0; i < 3652058; i += 97) {
    EXPECT_EQ( i, days_between(1/JAN/1, days_after(1/JAN/1, i)));
    EXPECT_EQ(-i, days_between(days_after(1/JAN/1, i), 1/JAN/1));

    EXPECT_EQ(-i, days_between(9999/DEC/31, days_before(9999/DEC/31, i)));
    EXPECT_EQ( i, days_between(days_before(9999/DEC/31, i), 9999/DEC/31));
  }
}

TEST(days_between, invalid) {
  EXPECT_THROW(days_between(1/JAN/1      , Date::INVALID), InvalidDateError);
  EXPECT_THROW(days_between(Date::INVALID, 1/JAN/1      ), InvalidDateError);
  EXPECT_THROW(days_between(1/JAN/1      , Date::MISSING), InvalidDateError);
  EXPECT_THROW(days_between(Date::MISSING, 1/JAN/1      ), InvalidDateError);
  EXPECT_THROW(days_between(Date::MISSING, Date::INVALID), InvalidDateError);
  EXPECT_THROW(days_between(Date::INVALID, Date::MISSING), InvalidDateError);

  EXPECT_THROW(days_between(Date16(2000/JAN/1), Date16::INVALID), InvalidDateError);
  EXPECT_THROW(days_between(Date16::INVALID, Date16(2000/JAN/1)), InvalidDateError);
  EXPECT_THROW(days_between(Date16(2000/JAN/1), Date16::MISSING), InvalidDateError);
  EXPECT_THROW(days_between(Date16::MISSING, Date16(2000/JAN/1)), InvalidDateError);
  EXPECT_THROW(days_between(Date16::MISSING, Date16::INVALID), InvalidDateError);
  EXPECT_THROW(days_between(Date16::INVALID, Date16::MISSING), InvalidDateError);
}

