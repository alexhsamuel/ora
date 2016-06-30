#include "gtest/gtest.h"
#include "cron.hh"

using namespace cron;
using namespace cron::time;

//------------------------------------------------------------------------------

TEST(compare, Time) {
  auto const i = Time::INVALID;
  auto const m = Time::MISSING;
  auto const t = Time::MIN;
  auto const u = Time::MAX;

  EXPECT_EQ( 0, safe::compare(i, i));
  EXPECT_EQ(-1, safe::compare(i, m));
  EXPECT_EQ(-1, safe::compare(i, t));
  EXPECT_EQ(-1, safe::compare(i, u));
  EXPECT_EQ( 1, safe::compare(m, i));
  EXPECT_EQ( 0, safe::compare(m, m));
  EXPECT_EQ(-1, safe::compare(m, t));
  EXPECT_EQ(-1, safe::compare(m, u));
  EXPECT_EQ( 1, safe::compare(t, i));
  EXPECT_EQ( 1, safe::compare(t, m));
  EXPECT_EQ( 0, safe::compare(t, t));
  EXPECT_EQ(-1, safe::compare(t, u));
  EXPECT_EQ( 1, safe::compare(u, i));
  EXPECT_EQ( 1, safe::compare(u, m));
  EXPECT_EQ( 1, safe::compare(u, t));
  EXPECT_EQ( 0, safe::compare(u, u));
}

TEST(equal, Time) {
  auto const i = Time::INVALID;
  auto const m = Time::MISSING;
  auto const t = Time::MIN;
  auto const u = Time::MAX;

  EXPECT_TRUE (safe::equal(i, i));
  EXPECT_FALSE(safe::equal(i, m));
  EXPECT_FALSE(safe::equal(i, t));
  EXPECT_FALSE(safe::equal(i, u));
  EXPECT_FALSE(safe::equal(m, i));
  EXPECT_TRUE (safe::equal(m, m));
  EXPECT_FALSE(safe::equal(m, t));
  EXPECT_FALSE(safe::equal(m, u));
  EXPECT_FALSE(safe::equal(t, i));
  EXPECT_FALSE(safe::equal(t, m));
  EXPECT_TRUE (safe::equal(t, t));
  EXPECT_FALSE(safe::equal(t, u));
  EXPECT_FALSE(safe::equal(u, i));
  EXPECT_FALSE(safe::equal(u, m));
  EXPECT_FALSE(safe::equal(u, t));
  EXPECT_TRUE (safe::equal(u, u));
}

TEST(before, Time) {
  auto const i = Time::INVALID;
  auto const m = Time::MISSING;
  auto const t = Time::MIN;
  auto const u = Time::MAX;

  EXPECT_FALSE(safe::before(i, i));
  EXPECT_TRUE (safe::before(i, m));
  EXPECT_TRUE (safe::before(i, t));
  EXPECT_TRUE (safe::before(i, u));
  EXPECT_FALSE(safe::before(m, i));
  EXPECT_FALSE(safe::before(m, m));
  EXPECT_TRUE (safe::before(m, t));
  EXPECT_TRUE (safe::before(m, u));
  EXPECT_FALSE(safe::before(t, i));
  EXPECT_FALSE(safe::before(t, m));
  EXPECT_FALSE(safe::before(t, t));
  EXPECT_TRUE (safe::before(t, u));
  EXPECT_FALSE(safe::before(u, i));
  EXPECT_FALSE(safe::before(u, m));
  EXPECT_FALSE(safe::before(u, t));
  EXPECT_FALSE(safe::before(u, u));
}

