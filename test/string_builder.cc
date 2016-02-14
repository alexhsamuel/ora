#include <string>

#include "gtest/gtest.h"
#include "string_builder.hh"

using namespace alxs;

using std::string;

//------------------------------------------------------------------------------
// Class StringBuilder
//------------------------------------------------------------------------------

TEST(StringBuilder, chars) {
  StringBuilder sb;
  sb << 'F' << 'o' << 'o';
  sb << '!';
  EXPECT_EQ(4u, sb.length());
  EXPECT_EQ("Foo!", sb.str());
}

TEST(StringBuilder, strings) {
  StringBuilder sb(1);
  sb << "Hello, world!" << ' ';
  sb << "This " << "is " << "a " << "test" << '.';
  EXPECT_EQ(29u, sb.length());
  EXPECT_EQ("Hello, world! This is a test.", sb.str());
}

TEST(StringBuilder, long0) {
  {
    StringBuilder sb;
    (sb << "x=").format(42);
    EXPECT_EQ("x=42", sb.str());
  }
  {
    StringBuilder sb(1);
    (sb << "x=").format(123456);
    EXPECT_EQ("x=123456", sb.str());
  }
  {
    StringBuilder sb;
    (sb << "x=").format(123456, 9);
    EXPECT_EQ("x=   123456", sb.str());
  }
  {
    StringBuilder sb;
    (sb << "x=").format(123456, 2);
    EXPECT_EQ("x=123456", sb.str());
  }
  {
    StringBuilder sb;
    (sb << "x=").format(123456, 9, '0');
    EXPECT_EQ("x=000123456", sb.str());
  }
}

