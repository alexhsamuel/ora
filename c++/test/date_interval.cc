#include "cron/date.hh"
#include "cron/date_interval.hh"
#include "cron/ez.hh"
#include "gtest/gtest.h"

using namespace alxs;
using namespace alxs::cron;
using namespace alxs::cron::ez;

using std::string;

//------------------------------------------------------------------------------
// Class DayInterval
//------------------------------------------------------------------------------

TEST(DayInterval, basic) {
  Date const date = 2013/JUL/22;
  EXPECT_EQ(2013/JUL/21, date -       DAY);
  EXPECT_EQ(2013/JUL/23, date +       DAY);
  EXPECT_EQ(2013/JUL/22, date +   0 * DAY);
  EXPECT_EQ(2013/JUL/25, date +   3 * DAY);
  EXPECT_EQ(2013/JUN/22, date -  30 * DAY);
  EXPECT_EQ(2013/JUN/22, date + -30 * DAY);
  EXPECT_EQ(2012/JUL/22, date - 365 * DAY);
  EXPECT_EQ(2014/JUL/22, 365 * DAY + date);
}

//------------------------------------------------------------------------------
// Class MonthInterval
//------------------------------------------------------------------------------

TEST(MonthInterval, basic) {
  Date const date = 2013/JUL/22;
  EXPECT_EQ(2013/JUL/22, date +  0 * MONTH);
  EXPECT_EQ(2013/AUG/22, date +      MONTH);
  EXPECT_EQ(2013/JUN/22, date -      MONTH);
  EXPECT_EQ(2013/NOV/22, date +  4 * MONTH);
  EXPECT_EQ(2014/JAN/22, date +  6 * MONTH);
  EXPECT_EQ(2013/MAR/22, date + -4 * MONTH);
  EXPECT_EQ(2014/JUL/22, date + 12 * MONTH);
  EXPECT_EQ(2014/JUL/22, date +      YEAR);
  EXPECT_EQ(2003/JUL/22, date - 10 * YEAR);
}

TEST(MonthInterval, special) {
  EXPECT_EQ(2013/FEB/28, 2013/JAN/31 + MONTH);
  EXPECT_EQ(2013/FEB/28, 2013/MAR/31 - MONTH);
  EXPECT_EQ(2013/FEB/28, 2013/MAR/30 - MONTH);
  EXPECT_EQ(2013/FEB/28, 2013/MAR/29 - MONTH);
  EXPECT_EQ(2013/FEB/28, 2013/MAR/28 - MONTH);
  EXPECT_EQ(2013/FEB/27, 2013/MAR/27 - MONTH);
  EXPECT_EQ(2013/FEB/28, 2012/FEB/29 + YEAR);
  EXPECT_EQ(2012/FEB/28, 2013/FEB/28 - YEAR);
  EXPECT_EQ(2016/FEB/29, 2012/FEB/29 + 4 * YEAR);
}
