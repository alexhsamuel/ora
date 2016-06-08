#include "gtest/gtest.h"
#include "cron/daytime.hh"
#include "cron/format.hh"

using namespace cron;
using namespace cron::daytime;

//------------------------------------------------------------------------------

bool
almost_equal(
  Daytime const d0,
  Daytime const d1,
  double const epsilon=1e-6)
{
  return std::abs((double) d0.get_offset() - d1.get_offset()) <= epsilon;
}


TEST(daytime_add, Daytime) {
  auto const d = from_hms(12, 30, 15);
  EXPECT_EQ(from_hms( 12, 30, 15.0), d +     0.0);
  EXPECT_TRUE(almost_equal(from_hms( 12, 30, 15.1), d +     0.1));
  EXPECT_EQ(from_hms( 12, 30, 16.0), d +     1.0);
  EXPECT_EQ(from_hms( 12, 30, 25.0), d +    10);
  EXPECT_EQ(from_hms( 12, 31, 55.0), d +   100);
  EXPECT_EQ(from_hms( 12, 46, 55.0), d +  1000);
  EXPECT_EQ(from_hms( 15, 16, 55.0), d + 10000);

  EXPECT_EQ(from_hms( 12, 30, 15.0), d +    -0.0);
  EXPECT_TRUE(almost_equal(from_hms( 12, 30, 14.9), d +    -0.1));
  EXPECT_EQ(from_hms( 12, 30, 14.0), d +    -1.0);
  EXPECT_EQ(from_hms( 12, 30,  5.0), d +   -10);
  EXPECT_EQ(from_hms( 12, 28, 35.0), d +  -100);
  EXPECT_EQ(from_hms( 12, 13, 35.0), d + -1000);
}

TEST(daytime_add, Daytime32) {
  Daytime32 const d = from_hms(12, 30, 15);
  EXPECT_EQ(from_hms<Daytime32>( 12, 30, 15.0), d +     0.0);
  EXPECT_EQ(from_hms<Daytime32>( 15, 16, 55.0), d + 10000);
  EXPECT_EQ(from_hms<Daytime32>( 12, 13, 35.0), d + -1000);
}

TEST(daytime_add, limits) {
  EXPECT_EQ(Daytime::MIDNIGHT, Daytime::MIDNIGHT + 0);
  EXPECT_TRUE(almost_equal(from_hms(23, 59, 59.99999), Daytime::MIDNIGHT + 86399.99999));
  EXPECT_EQ(Daytime::MIDNIGHT, from_hms(23, 59, 59.99999) + -86399.99999);
  EXPECT_EQ(Daytime::MIDNIGHT, from_hms(12, 30, 15) + -45015);
  EXPECT_EQ(from_hms(23, 59, 59), from_hms(12, 30, 15) + 41384);
}

TEST(daytime_add, range) {
  EXPECT_THROW(Daytime::MIDNIGHT + -1, DaytimeRangeError);
  EXPECT_THROW(Daytime::MIDNIGHT + 86400, DaytimeRangeError);
  EXPECT_THROW(from_hms(23, 59, 59) + 1, DaytimeRangeError);
  EXPECT_THROW(from_hms(23, 59, 59.999) + 0.001, DaytimeRangeError);
}

TEST(daytime_add, invalid) {
  EXPECT_THROW(Daytime::INVALID + 0, InvalidDaytimeError);
  EXPECT_THROW(Daytime::MISSING + 0, InvalidDaytimeError);
  EXPECT_THROW(Daytime32::INVALID + 0, InvalidDaytimeError);
  EXPECT_THROW(Daytime32::MISSING + 0, InvalidDaytimeError);
}

