#include <string>

#include "ora.hh"
#include "gtest/gtest.h"

using namespace ora::lib;
using namespace ora;
using namespace ora::ez;

using std::string;

//------------------------------------------------------------------------------
// Class Date
//------------------------------------------------------------------------------

TEST(Date, default_ctor) {
  Date const date;
  EXPECT_TRUE(date.is_invalid());
}

TEST(DATE, is_invalid) {
  EXPECT_TRUE (Date::INVALID.is_invalid());
  EXPECT_FALSE(Date::MISSING.is_invalid());
  EXPECT_FALSE(Date::MIN    .is_invalid());
  EXPECT_FALSE(Date::MAX    .is_invalid());
  EXPECT_FALSE((2016/JUN/10).is_invalid());
}

TEST(DATE, is_missing) {
  EXPECT_FALSE(Date::INVALID.is_missing());
  EXPECT_TRUE (Date::MISSING.is_missing());
  EXPECT_FALSE(Date::MIN    .is_missing());
  EXPECT_FALSE(Date::MAX    .is_missing());
  EXPECT_FALSE((2016/JUN/10).is_missing());
}

TEST(Date, range) {
  EXPECT_EQ("0001-01-01", to_string(Date::MIN));
  EXPECT_EQ("9999-12-31", to_string(Date::MAX));
}

TEST(Date, from_ymd) {
  auto const ymd = get_ymd(from_ymd(1973, 12, 3));
  EXPECT_EQ(1973, ymd.year);
  EXPECT_EQ(12, ymd.month);
  EXPECT_EQ(3, ymd.day);
}

TEST(Date, offsets) {
  EXPECT_EQ(Date::MIN, date::from_offset<Date>(Date::MIN.get_offset()));
  EXPECT_EQ(Date::MIN, from_datenum<Date>(Date::MIN.get_datenum()));

  EXPECT_EQ(Date::MAX, date::from_offset<Date>(Date::MAX.get_offset()));
  EXPECT_EQ(Date::MAX, from_datenum<Date>(Date::MAX.get_datenum()));

  auto const date1 = from_ymd(1600, 3, 1);
  EXPECT_EQ(date1, date::from_offset<Date>(date1.get_offset()));
  EXPECT_EQ(date1, from_datenum<Date>(date1.get_datenum()));

  auto const date2 = from_ymd(2000, 3, 1);
  EXPECT_EQ(date2, date::from_offset<Date>(date2.get_offset()));
  EXPECT_EQ(date2, from_datenum<Date>(date2.get_datenum()));
}

TEST(Date, shift) {
  auto const date0 = from_ymd(1973, 12, 3);
  auto const date1 = date0 + 1;
  auto const ymd = get_ymd(date1);
  EXPECT_EQ(ymd.year, 1973);
  EXPECT_EQ(ymd.month, 12);
  EXPECT_EQ(ymd.day, 4);
}

TEST(Date, is_valid) {
  EXPECT_TRUE ((1973/DEC/ 3).is_valid());
  EXPECT_TRUE (Date::MIN.is_valid());
  EXPECT_TRUE (Date::MAX.is_valid());
  EXPECT_FALSE(Date::MISSING.is_valid());
  EXPECT_FALSE(Date::INVALID.is_valid());
}

TEST(Date, invalid) {
  EXPECT_THROW(from_datenum<Date>(DATENUM_INVALID).is_invalid(), InvalidDateError);
  EXPECT_THROW(from_ymd(1973, 12,  32), InvalidDateError);
  EXPECT_THROW(from_ymd(1973, 11,  31), InvalidDateError);
  EXPECT_THROW(from_ymd(1973,  2,  29), InvalidDateError);
  EXPECT_THROW(from_ymd(1972,  2,  30), InvalidDateError);
  EXPECT_FALSE(from_ymd(1972,  2,  29).is_invalid());
  EXPECT_THROW(from_ymd(  -1,  1,   2), InvalidDateError);
  EXPECT_THROW(from_ymd(   0,  1,   1), InvalidDateError);
  EXPECT_FALSE(from_ymd(   1,  1,   1).is_invalid());
  EXPECT_FALSE(from_ymd(1000,  1,   2).is_invalid());
  EXPECT_THROW(from_ymd(1970, 13,   2), InvalidDateError);
  EXPECT_THROW(from_ymd(32000, 1,   2), InvalidDateError);
}


TEST(Date, range_error) {
  EXPECT_THROW(Date::MIN  -       1, DateRangeError);
  EXPECT_THROW(Date::MIN  +      -1, DateRangeError);
  EXPECT_THROW(Date::MAX  +       1, DateRangeError);
  EXPECT_THROW(Date::MAX  +   10000, DateRangeError);
  EXPECT_THROW(Date::MAX  + 1000000, DateRangeError);
}

TEST(Date, invalid_ymd) {
  EXPECT_THROW(get_ymd(Date::INVALID), InvalidDateError);
  EXPECT_THROW(Date::INVALID.get_datenum(), InvalidDateError);
}

TEST(Date, missing_ymd) {
  EXPECT_THROW(get_ymd(Date::MISSING), InvalidDateError);
  EXPECT_THROW(Date::MISSING.get_datenum(), InvalidDateError);
}

TEST(Date, weekday) {
  EXPECT_EQ(MONDAY,             get_weekday(   1/JAN/ 1));
  EXPECT_EQ(MONDAY,             get_weekday(1973/DEC/ 3));
  EXPECT_EQ(THURSDAY,           get_weekday(2016/MAR/24));
  EXPECT_EQ(FRIDAY,             get_weekday(9999/DEC/31));

  EXPECT_THROW(get_weekday(Date::INVALID), InvalidDateError);
  EXPECT_THROW(get_weekday(Date::MISSING), InvalidDateError);
}

TEST(Date, conversions) {
  EXPECT_EQ(from_ymd<Date16>(1973, 12, 3), Date16(1973/DEC/3));

  EXPECT_TRUE(date::nex::equal(Date16(Date::INVALID), Date16::INVALID));
  EXPECT_TRUE(date::nex::equal(Date16(Date::MISSING), Date16::MISSING));
  EXPECT_TRUE(date::nex::equal(Date(Date16::INVALID), Date::INVALID));
  EXPECT_TRUE(date::nex::equal(Date(Date16::MISSING), Date::MISSING));
}

TEST(Date, ostream) {
  Date const date = 1973/DEC/3;
  {
    std::stringstream ss;
    ss << date;
    EXPECT_EQ("1973-12-03", ss.str());
  }

  {
    DateFormat const format = "%0m/%0d/%y";
    std::stringstream ss;
    ss << format(date);
    EXPECT_EQ("12/3/73", ss.str());
  }

}

TEST(Date, from_iso_date) {
  EXPECT_EQ(from_iso_date<Date>("0001-01-01"),    1/JAN/ 1);
  EXPECT_EQ(from_iso_date<Date>("00010101"  ),    1/JAN/ 1);
  EXPECT_EQ(from_iso_date<Date>("1973-12-03"), 1973/DEC/ 3);
  EXPECT_EQ(from_iso_date<Date>("19731203"  ), 1973/DEC/ 3);
  EXPECT_EQ(from_iso_date<Date>("2016-02-29"), 2016/FEB/29);
  EXPECT_EQ(from_iso_date<Date>("20160229"  ), 2016/FEB/29);
  EXPECT_EQ(from_iso_date<Date>("9999-12-31"), 9999/DEC/31);
  EXPECT_EQ(from_iso_date<Date>("99991231"  ), 9999/DEC/31);
}

TEST(Date, from_iso_date_format_error) {
  EXPECT_THROW(from_iso_date<Date>(""), DateFormatError);
  EXPECT_THROW(from_iso_date<Date>("foobar"), DateFormatError);
  EXPECT_THROW(from_iso_date<Date>("2000-01-1"), DateFormatError);
  EXPECT_THROW(from_iso_date<Date>("2000-1-01"), DateFormatError);
  EXPECT_THROW(from_iso_date<Date>("500-1-1"), DateFormatError);
  EXPECT_THROW(from_iso_date<Date>("2000011"), DateFormatError);
  EXPECT_THROW(from_iso_date<Date>("2000101"), DateFormatError);
  EXPECT_THROW(from_iso_date<Date>("50011"), DateFormatError);
  EXPECT_THROW(from_iso_date<Date>("10000-01-01"), DateFormatError);
  EXPECT_THROW(
    from_iso_date<Date>("The quick brown fox jumped over the lazy dogs."), 
    DateFormatError);
}

TEST(Date, from_iso_date_invalid_error) {
  EXPECT_THROW(from_iso_date<Date>("2015-02-29"), InvalidDateError);
  EXPECT_THROW(from_iso_date<Date>("2015-03-32"), InvalidDateError);
  EXPECT_THROW(from_iso_date<Date>("2015-04-00"), InvalidDateError);
  EXPECT_THROW(from_iso_date<Date>("2015-13-01"), InvalidDateError);
}

TEST(Date, comparison) {
  Date const dates[] = {
    Date::INVALID, Date::MISSING, Date::MIN, 2016/JUN/7, 2016/JUL/4, Date::MAX
  };
  size_t const n = sizeof(dates) / sizeof(Date);
  for (size_t i0 = 0; i0 < n; ++i0)
    for (size_t i1 = 0; i1 < n; ++i1) {
      auto const d0 = dates[i0];
      auto const d1 = dates[i1];
      EXPECT_EQ(i0 == i1, d0 == d1);
      EXPECT_EQ(i0 != i1, d0 != d1);
      EXPECT_EQ(i0 <  i1, d0 <  d1);
      EXPECT_EQ(i0 <= i1, d0 <= d1);
      EXPECT_EQ(i0 >  i1, d0 >  d1);
      EXPECT_EQ(i0 >= i1, d0 >= d1);
    }
}

TEST(Date16, from_iso_date) {
  EXPECT_EQ(from_iso_date<Date16>("1970-01-01"), from_ymd<Date16>(1970, 1, 1));
  EXPECT_EQ(from_iso_date<Date16>("19700101"  ), from_ymd<Date16>(1970, 1, 1));
  EXPECT_EQ(from_iso_date<Date16>("2149-06-04"), from_ymd<Date16>(2149, 6, 4));
  EXPECT_EQ(from_iso_date<Date16>("21490604"  ), from_ymd<Date16>(2149, 6, 4));
}

TEST(Date16, from_iso_date_range_error) {
  EXPECT_THROW(from_iso_date<Date16>("0001-01-01"), DateRangeError);
  EXPECT_THROW(from_iso_date<Date16>("9999-12-31"), DateRangeError);
  EXPECT_THROW(from_iso_date<Date16>("00010101"), DateRangeError);
  EXPECT_THROW(from_iso_date<Date16>("99991231"), DateRangeError);
}

//------------------------------------------------------------------------------
// Class Date16
//------------------------------------------------------------------------------

TEST(Date16, default_ctor) {
  Date16 const date;
  EXPECT_TRUE(date.is_invalid());
}

TEST(Date16, range) {
  EXPECT_EQ(     0u, Date16::MIN.get_offset());
  EXPECT_EQ(719162u, Date16::MIN.get_datenum());

  EXPECT_EQ(from_ymd<Date16>(1970, 1, 1), Date16::MIN);
  EXPECT_EQ("1970-01-01", (string) DateFormat::ISO_CALENDAR_EXTENDED(Date16::MIN));
  EXPECT_EQ(from_ymd<Date16>(2149, 6, 4), Date16::MAX);
  EXPECT_EQ("2149-06-04", (string) DateFormat::ISO_CALENDAR_EXTENDED(Date16::MAX));
}

TEST(Date16, is_valid) {
  EXPECT_TRUE (Date16::MIN.is_valid());
  EXPECT_TRUE (Date16::MAX.is_valid());
  EXPECT_FALSE(Date16::MISSING.is_valid());
  EXPECT_FALSE(Date16::INVALID.is_valid());
  EXPECT_TRUE (from_ymd<Date16>(1970,  1, 1).is_valid());
  EXPECT_TRUE (from_ymd<Date16>(1970,  1, 2).is_valid());
  EXPECT_TRUE (from_ymd<Date16>(1973, 12, 3).is_valid());
}

TEST(Date16, from_ymd0) {
  Date16 const date = from_ymd(1970, 1, 1);
  EXPECT_EQ(0, date.get_offset());
  EXPECT_EQ(Date16(1970/JAN/ 1), date);

  auto const ymd = get_ymd(date);
  EXPECT_EQ(1970,   ymd.year);
  EXPECT_EQ(1,      ymd.month);
  EXPECT_EQ(1,      ymd.day);
}

TEST(Date16, from_ymd1) {
  Date16 const date = from_ymd<Date16>(1973, 12, 3);
  auto const ymd = get_ymd(date);
  EXPECT_EQ(1973,   ymd.year);
  EXPECT_EQ(12,     ymd.month);
  EXPECT_EQ(3,      ymd.day);
}

//------------------------------------------------------------------------------
// Easy literals.

TEST(MonthLiteral, basic) {
  EXPECT_EQ(from_ymd(1973, 1, 3), 1973/JAN/3);
}

