#include "ora.hh"
#include "gtest/gtest.h"

using namespace ora;

//------------------------------------------------------------------------------

TEST(weeday, encoding_cron) {
  for (Weekday w = WEEKDAY_MIN; w <= WEEKDAY_MAX; ++w) {
    EXPECT_TRUE(weekday::ENCODING_CRON::is_valid((int) w));
    EXPECT_EQ((int) w, weekday::ENCODING_CRON::encode(w));
    EXPECT_EQ(w, weekday::ENCODING_CRON::decode((int) w));
  }
}

TEST(weekday, encoding_iso) {
  for (int w = -2; w < 10; ++w)
    EXPECT_EQ(1 <= w && w <= 7, weekday::ENCODING_ISO::is_valid(w));

  EXPECT_EQ(1, weekday::ENCODING_ISO::encode(MONDAY));
  EXPECT_EQ(2, weekday::ENCODING_ISO::encode(TUESDAY));
  EXPECT_EQ(3, weekday::ENCODING_ISO::encode(WEDNESDAY));
  EXPECT_EQ(4, weekday::ENCODING_ISO::encode(THURSDAY));
  EXPECT_EQ(5, weekday::ENCODING_ISO::encode(FRIDAY));
  EXPECT_EQ(6, weekday::ENCODING_ISO::encode(SATURDAY));
  EXPECT_EQ(7, weekday::ENCODING_ISO::encode(SUNDAY));

  EXPECT_EQ(MONDAY   , weekday::ENCODING_ISO::decode(1));
  EXPECT_EQ(TUESDAY  , weekday::ENCODING_ISO::decode(2));
  EXPECT_EQ(WEDNESDAY, weekday::ENCODING_ISO::decode(3));
  EXPECT_EQ(THURSDAY , weekday::ENCODING_ISO::decode(4));
  EXPECT_EQ(FRIDAY   , weekday::ENCODING_ISO::decode(5));
  EXPECT_EQ(SATURDAY , weekday::ENCODING_ISO::decode(6));
  EXPECT_EQ(SUNDAY   , weekday::ENCODING_ISO::decode(7));
}

TEST(weekday, encoding_clib) {
  for (int w = -2; w < 10; ++w)
    EXPECT_EQ(0 <= w && w <= 6, weekday::ENCODING_CLIB::is_valid(w));

  EXPECT_EQ(0, weekday::ENCODING_CLIB::encode(SUNDAY));
  EXPECT_EQ(1, weekday::ENCODING_CLIB::encode(MONDAY));
  EXPECT_EQ(2, weekday::ENCODING_CLIB::encode(TUESDAY));
  EXPECT_EQ(3, weekday::ENCODING_CLIB::encode(WEDNESDAY));
  EXPECT_EQ(4, weekday::ENCODING_CLIB::encode(THURSDAY));
  EXPECT_EQ(5, weekday::ENCODING_CLIB::encode(FRIDAY));
  EXPECT_EQ(6, weekday::ENCODING_CLIB::encode(SATURDAY));

  EXPECT_EQ(SUNDAY   , weekday::ENCODING_CLIB::decode(0));
  EXPECT_EQ(MONDAY   , weekday::ENCODING_CLIB::decode(1));
  EXPECT_EQ(TUESDAY  , weekday::ENCODING_CLIB::decode(2));
  EXPECT_EQ(WEDNESDAY, weekday::ENCODING_CLIB::decode(3));
  EXPECT_EQ(THURSDAY , weekday::ENCODING_CLIB::decode(4));
  EXPECT_EQ(FRIDAY   , weekday::ENCODING_CLIB::decode(5));
  EXPECT_EQ(SATURDAY , weekday::ENCODING_CLIB::decode(6));
}

