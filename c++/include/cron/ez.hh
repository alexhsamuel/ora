#pragma once

#include "cron/date.hh"
#include "cron/date_functions.hh"
#include "cron/types.hh"

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

    constexpr 
    YearMonthLiteral(
      Year const year, 
      Month const month) 
    : year_(year), 
      month_(month) 
    {}
    
    date::Date 
    operator/(
      Day day) 
      const 
    { 
      return date::from_ymd(year_, month_, day); 
    }

  private:

    Year year_;
    Month month_;

  };

public:

  constexpr MonthLiteral(Month month) : month_(month) {}
  constexpr YearMonthLiteral with_year(Year year) const { return YearMonthLiteral(year, month_); }
  
  explicit operator Month() const { return month_; }

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


inline bool
operator==(
  MonthLiteral const month0,
  MonthLiteral const month1)
{
  return (Month) month0 == (Month) month1;
}


inline bool
operator==(
  Month const month0,
  MonthLiteral const month1)
{
  return month0 == (Month) month1;
}


inline bool
operator==(
  MonthLiteral const month0,
  Month const month1)
{
  return (Month) month0 == month1;
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


