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

TEST(first_of_month, basic) {
  EXPECT_EQ(ymd_to_datenum(2022,  1,  1), first_of_month(2022,  1));
  EXPECT_EQ(ymd_to_datenum(2022,  2,  1), first_of_month(2022,  2));
  EXPECT_EQ(ymd_to_datenum(2022,  3,  1), first_of_month(2022,  3));
  EXPECT_EQ(ymd_to_datenum(2022,  4,  1), first_of_month(2022,  4));
  EXPECT_EQ(ymd_to_datenum(2022,  5,  1), first_of_month(2022,  5));
  EXPECT_EQ(ymd_to_datenum(2022,  6,  1), first_of_month(2022,  6));
  EXPECT_EQ(ymd_to_datenum(2022,  7,  1), first_of_month(2022,  7));
  EXPECT_EQ(ymd_to_datenum(2022,  8,  1), first_of_month(2022,  8));
  EXPECT_EQ(ymd_to_datenum(2022,  9,  1), first_of_month(2022,  9));
  EXPECT_EQ(ymd_to_datenum(2022, 10,  1), first_of_month(2022, 10));
  EXPECT_EQ(ymd_to_datenum(2022, 11,  1), first_of_month(2022, 11));
  EXPECT_EQ(ymd_to_datenum(2022, 12,  1), first_of_month(2022, 12));
}

TEST(last_of_month, basic) {
  EXPECT_EQ(ymd_to_datenum(2022,  1, 31), last_of_month(2022,  1));
  EXPECT_EQ(ymd_to_datenum(2022,  2, 28), last_of_month(2022,  2));
  EXPECT_EQ(ymd_to_datenum(2022,  3, 31), last_of_month(2022,  3));
  EXPECT_EQ(ymd_to_datenum(2022,  4, 30), last_of_month(2022,  4));
  EXPECT_EQ(ymd_to_datenum(2022,  5, 31), last_of_month(2022,  5));
  EXPECT_EQ(ymd_to_datenum(2022,  6, 30), last_of_month(2022,  6));
  EXPECT_EQ(ymd_to_datenum(2022,  7, 31), last_of_month(2022,  7));
  EXPECT_EQ(ymd_to_datenum(2022,  8, 31), last_of_month(2022,  8));
  EXPECT_EQ(ymd_to_datenum(2022,  9, 30), last_of_month(2022,  9));
  EXPECT_EQ(ymd_to_datenum(2022, 10, 31), last_of_month(2022, 10));
  EXPECT_EQ(ymd_to_datenum(2022, 11, 30), last_of_month(2022, 11));
  EXPECT_EQ(ymd_to_datenum(2022, 12, 31), last_of_month(2022, 12));

  EXPECT_EQ(ymd_to_datenum(2022,  2, 28), last_of_month(2022,  2));
  EXPECT_EQ(ymd_to_datenum(2023,  2, 28), last_of_month(2023,  2));
  EXPECT_EQ(ymd_to_datenum(2024,  2, 29), last_of_month(2024,  2));
  EXPECT_EQ(ymd_to_datenum(2025,  2, 28), last_of_month(2025,  2));
}

TEST(weekday_of_month, basic) {
  EXPECT_EQ(ymd_to_datenum(2022,  4,  1), weekday_of_month(2022, 4, 1, FRIDAY));
  EXPECT_EQ(ymd_to_datenum(2022,  4,  2), weekday_of_month(2022, 4, 1, SATURDAY));
  EXPECT_EQ(ymd_to_datenum(2022,  4,  3), weekday_of_month(2022, 4, 1, SUNDAY));
  EXPECT_EQ(ymd_to_datenum(2022,  4,  4), weekday_of_month(2022, 4, 1, MONDAY));
  EXPECT_EQ(ymd_to_datenum(2022,  4,  5), weekday_of_month(2022, 4, 1, TUESDAY));
  EXPECT_EQ(ymd_to_datenum(2022,  4, 12), weekday_of_month(2022, 4, 2, TUESDAY));
  EXPECT_EQ(ymd_to_datenum(2022,  4, 19), weekday_of_month(2022, 4, 3, TUESDAY));
  EXPECT_EQ(ymd_to_datenum(2022,  4, 26), weekday_of_month(2022, 4, 4, TUESDAY));
  EXPECT_EQ(ymd_to_datenum(2022,  4,  5), weekday_of_month(2022, 4,-4, TUESDAY));
  EXPECT_EQ(ymd_to_datenum(2022,  4, 12), weekday_of_month(2022, 4,-3, TUESDAY));
  EXPECT_EQ(ymd_to_datenum(2022,  4, 19), weekday_of_month(2022, 4,-2, TUESDAY));
  EXPECT_EQ(ymd_to_datenum(2022,  4, 26), weekday_of_month(2022, 4,-1, TUESDAY));
  EXPECT_EQ(ymd_to_datenum(2022,  4, 29), weekday_of_month(2022, 4, 5, FRIDAY));
}

