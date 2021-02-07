#include "ora.hh"
#include "gtest/gtest.h"

using namespace ora;

//------------------------------------------------------------------------------

TEST(PosixTz, parse) {
  using Transition = PosixTz::Transition;

  auto const ptz = parse_posix_time_zone("EST5EDT,M3.2.0,M11.1.0");
  EXPECT_EQ(ptz.std.abbreviation, "EST");
  EXPECT_EQ(ptz.std.offset, -18000);
  EXPECT_EQ(ptz.dst.abbreviation, "EDT");
  EXPECT_EQ(ptz.dst.offset, -14400);
  EXPECT_EQ(ptz.start.type, Transition::Type::GREGORIAN);
  EXPECT_EQ(ptz.start.spec.gregorian.month, 3);
  EXPECT_EQ(ptz.start.spec.gregorian.week, 2);
  EXPECT_EQ(ptz.start.spec.gregorian.weekday, 0);
  EXPECT_EQ(ptz.end.type, Transition::Type::GREGORIAN);
  EXPECT_EQ(ptz.end.spec.gregorian.month, 11);
  EXPECT_EQ(ptz.end.spec.gregorian.week, 1);
  EXPECT_EQ(ptz.end.spec.gregorian.weekday, 0);
}

