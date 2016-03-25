#pragma once

#include "aslib/math.hh"
#include "cron/types.hh"

namespace cron {

//------------------------------------------------------------------------------
// Construction functions
//------------------------------------------------------------------------------

/*
 * Creates a date from its (date class-specific) offset.
 *
 * Returns an invalid date if the offset is not valid.
 */
template<typename DATE>
inline DATE
from_offset(
  typename DATE::Offset const offset)
{
  return 
      offset_is_valid<DATE::TRAITS>(offset)
    ? DATE::from_offset(offset)
    : DATE::INVALID;
}


template<typename DATE>
inline DATE
from_datenum(
  Datenum const datenum)
{
  return DATE(datenum_to_offset<DATE::TRAITS>(datenum));
}


template<typename DATE>
inline DATE
from_ordinal_date(
  Year const year,
  Ordinal const ordinal)
{
  return from_datenum<DATE>(ordinal_date_to_datenum(year, ordinal));
}


template<typename DATE>
inline DATE
from_ymd(
  Year const year,
  Month const month,
  Day const day)
{
  return DATE(datenum_to_offset<TRAITS>(ymd_to_datenum(year, month, day)));
}


template<typename DATE>
inline DATE
from_week_date(
  Year const week_year,
  Week const week,
  Weekday const weekday)
{
  return
      week_date_is_valid(week_year, week, weekday)
    ? DATE(datenum_to_offset<TRAITS>(
        week_date_to_datenum(week_year, week, weekday)))
    : DATE::INVALID;
}


//------------------------------------------------------------------------------
// Arithmetic
//------------------------------------------------------------------------------

template<typename DATE>
inline DATE
add(
  DATE const date,
  int shift)
{
  return 
      date.is_valid()
    ? from_offset<DATE>(date.get_offset() + shift)
    : date;
}  


template<typename DATE>
inline DATE
subtract(
  DATE const date,
  int shift)
{
  return add(date, -shift);
}  


template<typename DATE>
inline int
subtract(
  DATE const date0,
  DATE const date1)
{
  return
      date0.is_valid() && date1.is_valid()
    ? (int) date0.get_offset() - date1.get_offset()
    : std::numeric_limits<int>::min();
}


//------------------------------------------------------------------------------

}  // namespace cron


