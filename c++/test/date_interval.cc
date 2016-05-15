#include "cron/date.hh"
#include "cron/date_interval.hh"
#include "cron/ez.hh"
#include "gtest/gtest.h"

using namespace aslib;
using namespace cron;
using namespace cron::date;
using namespace cron::ez;

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

