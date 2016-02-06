#pragma once

#include "cron/date.hh"
#include "cron/types.hh"

namespace alxs {
namespace cron {
namespace ez {

//------------------------------------------------------------------------------
// Syntactic sugar for date literals.

// FIXME: Is this even a good idea?

namespace {

class MonthLiteral
{
private:

  class YearMonthLiteral
  {
  public:

    constexpr YearMonthLiteral(Year year, Month month) : year_(year), month_(month) {}
    Date operator/(Day day) const { return Date(year_, month_ - 1, day - 1); }

  private:

    Year year_;
    Month month_;

  };

public:

  constexpr MonthLiteral(Month month) : month_(month) {}
  constexpr YearMonthLiteral with_year(Year year) const { return YearMonthLiteral(year, month_); }
  
private:

  Month month_;

  friend YearMonthLiteral operator/(Year year, MonthLiteral const&);

};


inline
MonthLiteral::YearMonthLiteral
operator/(
  Year year,
  MonthLiteral const& month)
{
  return month.with_year(year);
}


}  // anonymous namespace

MonthLiteral constexpr JAN = MonthLiteral( 1);
MonthLiteral constexpr FEB = MonthLiteral( 2);
MonthLiteral constexpr MAR = MonthLiteral( 3);
MonthLiteral constexpr APR = MonthLiteral( 4);
MonthLiteral constexpr MAY = MonthLiteral( 5);
MonthLiteral constexpr JUN = MonthLiteral( 6);
MonthLiteral constexpr JUL = MonthLiteral( 7);
MonthLiteral constexpr AUG = MonthLiteral( 8);
MonthLiteral constexpr SEP = MonthLiteral( 9);
MonthLiteral constexpr OCT = MonthLiteral(10);
MonthLiteral constexpr NOV = MonthLiteral(11);
MonthLiteral constexpr DEC = MonthLiteral(12);

//------------------------------------------------------------------------------

}  // namespace ez
}  // namespace cron
}  // namespace alxs

