#include "cron/time.hh"
#include "cron/format.hh"
#include "cron/time_interval.hh"
#include "gtest/gtest.h"

using namespace aslib;
using namespace cron;
using namespace cron::time;

//------------------------------------------------------------------------------
// Class TimeInterval
//------------------------------------------------------------------------------

TEST(TimeInterval, basic) {
  auto tz = get_time_zone("US/Eastern");
  Time const time(2013, 8, 25, 23, 20, 15.5, *tz);

  EXPECT_EQ(Time(2013, 8, 25, 23, 20, 15.5000000, *tz), time + 0 * NANOSECOND);
  EXPECT_EQ(Time(2013, 8, 25, 23, 20, 15.5000000, *tz), time + 0 * SECOND);
  EXPECT_EQ(Time(2013, 8, 25, 23, 20, 15.5000001, *tz), time + 100 * NANOSECOND);
  EXPECT_EQ(Time(2013, 8, 25, 23, 20, 15.5000010, *tz), time + MICROSECOND);
  EXPECT_EQ(Time(2013, 8, 25, 23, 20, 15.5000042, *tz), time + 4.2 * MICROSECOND);
  EXPECT_EQ(Time(2013, 8, 25, 23, 20, 15.5000170, *tz), time + 17 * MICROSECOND);
  EXPECT_EQ(Time(2013, 8, 25, 23, 20, 15.5034000, *tz), time + 3400 * MICROSECOND);
  EXPECT_EQ(Time(2013, 8, 25, 23, 20, 15.5034000, *tz), time + 3.4 * MILLISECOND);
  EXPECT_EQ(Time(2013, 8, 25, 23, 20, 15.5034000, *tz), time + 0.0034 * SECOND);
  EXPECT_EQ(Time(2013, 8, 25, 23, 20, 16.5000000, *tz), time + SECOND);
  EXPECT_EQ(Time(2013, 8, 25, 23, 21,  0.0000000, *tz), time + 44.5 * SECOND);
}

