#pragma once

#include "cron/math.hh"
#include "cron/types.hh"

namespace alxs {
namespace cron {

//------------------------------------------------------------------------------
// Helper functions
//------------------------------------------------------------------------------

/*
 * If `datenum` is valid and can be converted to a valid offset, returns that;
 * otherwise the invalid offset.
 */
template<class TRAITS>
inline Offset
datenum_to_offset(
  Datenum const datenum)
{
  Datenum offset;
  return 
         datenum_is_valid(datenum)
      && !sub_overflow(datenum, (Datenum) TRAITS::base, offset)
      && in_range(TRAITS::min, offset, TRAITS::max)
    ? offset
    : TRAITS::invalid;
}


template<class TRAITS>
inline Offset
ymd_to_offset(
  Year const year,
  Month const month,
  Day const day)
{
  return
      ymd_is_valid(year, month, day)
    ? datenum_to_offset<TRAITS>(ymd_to_datenum(year, month, day))
    : TRAITS::invalid;
}


//------------------------------------------------------------------------------

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
date_from_week_date(
  Year const week_year,
  Week const week,
  Weekday const weekday)
{
  return
      week_date_is_valid(week_year, week, weekday)
    ? DATE(datenum_to_offset<TRAITS>(week_date_to_datenum(week_year, week, weekday)))
    : DATE::INVALID;
}


//------------------------------------------------------------------------------

}  // namespace cron
}  // namespace alxs


