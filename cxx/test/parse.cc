#include "ora.hh"
#include "gtest/gtest.h"

using namespace ora;

//------------------------------------------------------------------------------
// parse_date
//------------------------------------------------------------------------------

TEST(parse_date, basic) {
  EXPECT_EQ(from_ymd(2020,  1, 29), date::parse("%Y-%m-%d", "2020-01-29"));
  EXPECT_EQ(from_ymd(2020,  1, 29), date::parse("%Y%m%d", "20200129"));
  EXPECT_EQ(from_ymd(2020,  1, 29), date::parse("%D", "2020-01-29"));
}

TEST(parse_date, abbrev) {
  EXPECT_EQ(from_ymd(2020,  1, 29), date::parse("%~D", "20200129"));
}

