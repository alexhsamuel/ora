#include "cron/time.hh"
#include "cron/time_zone.hh"
#include "gtest/gtest.h"

using namespace aslib;
using namespace cron;

//------------------------------------------------------------------------------
// Class TimeZone.

TEST(TimeZone, get_time_zone) {
  auto const tz = get_time_zone("US/Eastern");
  ASSERT_EQ("US/Eastern", tz->get_name());
}

TEST(TimeZone, get_parts) {
  // 2013 July 26 14:26:38 EDT.
  auto const time = time::Unix64Time::from_offset(1374863198);

  auto const tz = get_time_zone("US/Eastern");
  auto const parts = tz->get_parts(time);
  EXPECT_TRUE(parts.is_dst);
  EXPECT_EQ(-14400, parts.offset);
  EXPECT_STREQ("EDT", parts.abbreviation);
}

TEST(TimeZone, get_parts_local) {
  auto const tz = get_time_zone("US/Eastern");

  auto const parts0 = tz->get_parts_local(1362880799);
  EXPECT_FALSE(parts0.is_dst);
  EXPECT_EQ(-18000, parts0.offset);

  EXPECT_THROW(tz->get_parts_local(1362880800), NonexistentDateDaytime);

  auto const parts2 = tz->get_parts_local(1362884400);
  EXPECT_TRUE(parts2.is_dst);
  EXPECT_EQ(-14400, parts2.offset);
}

// FIXME: Not general.
TEST(TimeZone, DISABLED_get_system_time_zone) {
  auto const tz = get_system_time_zone();
  EXPECT_EQ("America/New_York", tz->get_name());

  auto const time = time::Unix64Time::from_offset(1374863198);
  auto const parts = tz->get_parts(time);
  EXPECT_TRUE(parts.is_dst);
  EXPECT_EQ(-14400, parts.offset);
  EXPECT_STREQ("EDT", parts.abbreviation);
}  

TEST(TimeZone, get_display_time_zone) {
  set_display_time_zone("US/Eastern");
  EXPECT_EQ("US/Eastern", get_display_time_zone()->get_name());
  set_display_time_zone("US/Pacific");
  EXPECT_EQ("US/Pacific", get_display_time_zone()->get_name());
  set_display_time_zone("US/Eastern");
  EXPECT_EQ("US/Eastern", get_display_time_zone()->get_name());
}  

