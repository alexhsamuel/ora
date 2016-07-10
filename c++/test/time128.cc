#include "cron.hh"
#include "gtest/gtest.h"

using namespace aslib;
using namespace cron;
using namespace cron::ez;

//------------------------------------------------------------------------------

TEST(Time128, convert_to) {
  Time const t0 = from_utc_parts(2016, 5, 27, 16, 13, 26.577521);
  Time128 const t = t0;
  EXPECT_TRUE(t.is_valid());

  auto parts = get_parts(t, UTC);
  EXPECT_EQ(2016, parts.date.year);
  EXPECT_EQ(   5, parts.date.month);
  EXPECT_EQ(  27, parts.date.day);
  EXPECT_EQ(  16, parts.daytime.hour);
  EXPECT_EQ(  13, parts.daytime.minute);
  EXPECT_TRUE(std::abs(parts.daytime.second - 26.577521) < 1e-6);
}

