#include "ora.hh"
#include "ora/time_interval.hh"
#include "gtest/gtest.h"

using namespace aslib;
using namespace ora;

//------------------------------------------------------------------------------
// Class TimeInterval
//------------------------------------------------------------------------------

TEST(TimeInterval, basic) {
  auto tz = get_time_zone("US/Eastern");
  auto const time = from_local_parts(2013, 9, 26, 23, 20, 15.5, *tz);

  EXPECT_EQ(from_local_parts(2013, 9, 26, 23, 20, 15.5000000, *tz), time + 0 * NANOSECOND);
  EXPECT_EQ(from_local_parts(2013, 9, 26, 23, 20, 15.5000000, *tz), time + 0 * SECOND);
  EXPECT_EQ(from_local_parts(2013, 9, 26, 23, 20, 15.5000001, *tz), time + 100 * NANOSECOND);
  EXPECT_EQ(from_local_parts(2013, 9, 26, 23, 20, 15.5000010, *tz), time + MICROSECOND);
  EXPECT_EQ(from_local_parts(2013, 9, 26, 23, 20, 15.5000042, *tz), time + 4.2 * MICROSECOND);
  EXPECT_EQ(from_local_parts(2013, 9, 26, 23, 20, 15.5000170, *tz), time + 17 * MICROSECOND);
  EXPECT_EQ(from_local_parts(2013, 9, 26, 23, 20, 15.5034000, *tz), time + 3400 * MICROSECOND);
  EXPECT_EQ(from_local_parts(2013, 9, 26, 23, 20, 15.5034000, *tz), time + 3.4 * MILLISECOND);
  EXPECT_EQ(from_local_parts(2013, 9, 26, 23, 20, 15.5034000, *tz), time + 0.0034 * SECOND);
  EXPECT_EQ(from_local_parts(2013, 9, 26, 23, 20, 16.5000000, *tz), time + SECOND);
  EXPECT_EQ(from_local_parts(2013, 9, 26, 23, 21,  0.0000000, *tz), time + 44.5 * SECOND);
}

