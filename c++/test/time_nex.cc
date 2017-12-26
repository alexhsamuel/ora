#include "gtest/gtest.h"
#include "ora.hh"

using namespace ora;

//------------------------------------------------------------------------------

TEST(compare, Time) {
  auto const i = Time::INVALID;
  auto const m = Time::MISSING;
  auto const t = Time::MIN;
  auto const u = Time::MAX;

  EXPECT_EQ( 0, time::nex::compare(i, i));
  EXPECT_EQ(-1, time::nex::compare(i, m));
  EXPECT_EQ(-1, time::nex::compare(i, t));
  EXPECT_EQ(-1, time::nex::compare(i, u));
  EXPECT_EQ( 1, time::nex::compare(m, i));
  EXPECT_EQ( 0, time::nex::compare(m, m));
  EXPECT_EQ(-1, time::nex::compare(m, t));
  EXPECT_EQ(-1, time::nex::compare(m, u));
  EXPECT_EQ( 1, time::nex::compare(t, i));
  EXPECT_EQ( 1, time::nex::compare(t, m));
  EXPECT_EQ( 0, time::nex::compare(t, t));
  EXPECT_EQ(-1, time::nex::compare(t, u));
  EXPECT_EQ( 1, time::nex::compare(u, i));
  EXPECT_EQ( 1, time::nex::compare(u, m));
  EXPECT_EQ( 1, time::nex::compare(u, t));
  EXPECT_EQ( 0, time::nex::compare(u, u));
}

TEST(equal, Time) {
  auto const i = Time::INVALID;
  auto const m = Time::MISSING;
  auto const t = Time::MIN;
  auto const u = Time::MAX;

  EXPECT_TRUE (time::nex::equal(i, i));
  EXPECT_FALSE(time::nex::equal(i, m));
  EXPECT_FALSE(time::nex::equal(i, t));
  EXPECT_FALSE(time::nex::equal(i, u));
  EXPECT_FALSE(time::nex::equal(m, i));
  EXPECT_TRUE (time::nex::equal(m, m));
  EXPECT_FALSE(time::nex::equal(m, t));
  EXPECT_FALSE(time::nex::equal(m, u));
  EXPECT_FALSE(time::nex::equal(t, i));
  EXPECT_FALSE(time::nex::equal(t, m));
  EXPECT_TRUE (time::nex::equal(t, t));
  EXPECT_FALSE(time::nex::equal(t, u));
  EXPECT_FALSE(time::nex::equal(u, i));
  EXPECT_FALSE(time::nex::equal(u, m));
  EXPECT_FALSE(time::nex::equal(u, t));
  EXPECT_TRUE (time::nex::equal(u, u));
}

TEST(before, Time) {
  auto const i = Time::INVALID;
  auto const m = Time::MISSING;
  auto const t = Time::MIN;
  auto const u = Time::MAX;

  EXPECT_FALSE(time::nex::before(i, i));
  EXPECT_TRUE (time::nex::before(i, m));
  EXPECT_TRUE (time::nex::before(i, t));
  EXPECT_TRUE (time::nex::before(i, u));
  EXPECT_FALSE(time::nex::before(m, i));
  EXPECT_FALSE(time::nex::before(m, m));
  EXPECT_TRUE (time::nex::before(m, t));
  EXPECT_TRUE (time::nex::before(m, u));
  EXPECT_FALSE(time::nex::before(t, i));
  EXPECT_FALSE(time::nex::before(t, m));
  EXPECT_FALSE(time::nex::before(t, t));
  EXPECT_TRUE (time::nex::before(t, u));
  EXPECT_FALSE(time::nex::before(u, i));
  EXPECT_FALSE(time::nex::before(u, m));
  EXPECT_FALSE(time::nex::before(u, t));
  EXPECT_FALSE(time::nex::before(u, u));
}

