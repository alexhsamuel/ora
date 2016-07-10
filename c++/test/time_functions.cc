#include "cron.hh"
#include "gtest/gtest.h"

using namespace cron;

//------------------------------------------------------------------------------

TEST(compare, Time) {
  EXPECT_THROW(compare(Time::INVALID, Time::MIN), InvalidTimeError);
  EXPECT_THROW(compare(Time::MISSING, Time::MAX), InvalidTimeError);
  EXPECT_THROW(compare(Time::MAX, Time::INVALID), InvalidTimeError);
  EXPECT_THROW(compare(Time::MIN, Time::MISSING), InvalidTimeError);

  EXPECT_EQ( 0, compare(Time::MIN, Time::MIN));
  EXPECT_EQ(-1, compare(Time::MIN, Time::MAX));
  EXPECT_EQ( 1, compare(Time::MAX, Time::MIN));
  EXPECT_EQ( 0, compare(Time::MAX, Time::MAX));
}

TEST(equal, Time) {
  EXPECT_THROW(equal(Time::INVALID, Time::MIN), InvalidTimeError);
  EXPECT_THROW(equal(Time::MISSING, Time::MAX), InvalidTimeError);
  EXPECT_THROW(equal(Time::MAX, Time::INVALID), InvalidTimeError);
  EXPECT_THROW(equal(Time::MIN, Time::MISSING), InvalidTimeError);

  EXPECT_TRUE (equal(Time::MIN, Time::MIN));
  EXPECT_FALSE(equal(Time::MIN, Time::MAX));
  EXPECT_FALSE(equal(Time::MAX, Time::MIN));
  EXPECT_TRUE (equal(Time::MAX, Time::MAX));
}

TEST(before, Time) {
  EXPECT_THROW(before(Time::INVALID, Time::MIN), InvalidTimeError);
  EXPECT_THROW(before(Time::MISSING, Time::MAX), InvalidTimeError);
  EXPECT_THROW(before(Time::MAX, Time::INVALID), InvalidTimeError);
  EXPECT_THROW(before(Time::MIN, Time::MISSING), InvalidTimeError);

  EXPECT_FALSE(before(Time::MIN, Time::MIN));
  EXPECT_TRUE (before(Time::MIN, Time::MAX));
  EXPECT_FALSE(before(Time::MAX, Time::MIN));
  EXPECT_FALSE(before(Time::MAX, Time::MAX));
}

