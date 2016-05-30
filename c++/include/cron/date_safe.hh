#pragma once

#include <string>

#include "aslib/math.hh"
#include "cron/date.hh"
#include "cron/date_math.hh"
#include "cron/types.hh"

namespace cron {
namespace date {
namespace safe {

//------------------------------------------------------------------------------
// Forward declarations
//------------------------------------------------------------------------------

template<class DATE> inline DATE from_offset(typename DATE::Offset) noexcept;
template<class DATE> inline DATE from_ymd(YmdDate) noexcept;

//------------------------------------------------------------------------------
// Factory functions
//------------------------------------------------------------------------------

template<class DATE>
inline DATE
from_datenum(
  Datenum const datenum)
  noexcept
{
  using Offset = typename DATE::Offset;

  if (datenum_is_valid(datenum)) {
    auto offset = (long) datenum - (long) DATE::Traits::base;
    return 
        in_range((long) DATE::Traits::min, offset, (long) DATE::Traits::max)
      ? DATE((Offset) offset)
      : DATE::INVALID;
  }
  else
    return DATE::INVALID;
}


template<class DATE>
inline DATE
from_iso_date(
  std::string const& date)
  noexcept
{
  return from_ymd<DATE>(parse_iso_date(date));
}


/*
 * Creates a date from its (date class-specific) offset.
 *
 * Returns an invalid date if the offset is not valid.
 */
template<class DATE>
inline DATE
from_offset(
  typename DATE::Offset const offset)
  noexcept
{
  return 
      in_range(DATE::Traits::min, offset, DATE::Traits::max)
    ? DATE(offset)
    : DATE::INVALID;
}


template<class DATE>
inline DATE
from_ordinal_date(
  Year const year,
  Ordinal const ordinal)
  noexcept
{
  return 
      ordinal_date_is_valid(year, ordinal)
    ? from_datenum<DATE>(ordinal_date_to_datenum(year, ordinal))
    : DATE::INVALID;
}


template<class DATE>
DATE
from_week_date(
  Year const week_year,
  Week const week,
  Weekday const weekday)
  noexcept
{
  return 
      week_date_is_valid(week_year, week, weekday)
    ? from_datenum<DATE>(week_date_to_datenum(week_year, week, weekday))
    : DATE::INVALID;
}


template<class DATE>
inline DATE
from_ymd(
  Year const year,
  Month const month,
  Day const day)
  noexcept
{
  return 
      ymd_is_valid(year, month, day) 
    ? from_datenum<DATE>(ymd_to_datenum(year, month, day))
    : DATE::INVALID;
}


template<class DATE>
inline DATE
from_ymd(
  YmdDate const date)
  noexcept
{
  return from_ymd<DATE>(date.year, date.month, date.day);
}


template<class DATE>
inline DATE
from_ymdi(
  int const ymdi)
  noexcept
{
  return 
      ymdi_is_valid(ymdi)
    ? from_datenum<DATE>(ymdi_to_datenum(ymdi))
    : DATE::INVALID;
}


//------------------------------------------------------------------------------
// Accessors
//------------------------------------------------------------------------------

template<class DATE>
inline Datenum
get_datenum(
  DATE const date)
  noexcept
{
  return date.is_valid() ? date.get_datenum() : DATENUM_INVALID;
}


template<class DATE>
inline OrdinalDate
get_ordinal_date(
  DATE const date)
  noexcept
{
  return 
      date.is_valid() 
    ? datenum_to_ordinal_date(date.get_datenum())
    : OrdinalDate{};  // invalid
}


template<class DATE>
inline WeekDate
get_week_date(
  DATE const date)
  noexcept
{
  return 
      date.is_valid() 
    ? datenum_to_week_date(date.get_datenum()) 
    : WeekDate{};  // invalid
}


template<class DATE>
inline Weekday
get_weekday(
  DATE const date)
  noexcept
{
  return 
      date.is_valid() 
    ? cron::get_weekday(date.get_datenum()) 
    : WEEKDAY_INVALID;
}


template<class DATE>
inline YmdDate
get_ymd(
  DATE const date)
  noexcept
{
  return 
      date.is_valid()
    ? datenum_to_ymd(date.get_datenum())
    : YmdDate{};  // invalid
}


/*
 * Returns the int-encoded YYYYMMDD representation of a date.
 *
 * Returns YMDI_INVALID if `date` is not valid.
 */
template<class DATE>
inline int
get_ymdi(
  DATE const date)
  noexcept
{
  return 
      date.is_valid() 
    ? datenum_to_ymdi(date.get_datenum()) 
    : YMDI_INVALID;
}


template<class DATE> inline Year get_year(DATE const date) noexcept 
  { return safe::get_ordinal_date(date).year;  }
template<class DATE> inline Month get_month(DATE const date) noexcept 
  { return safe::get_ymd(date).month; }
template<class DATE> inline Ordinal get_ordinal(DATE const date) noexcept
  { return safe::get_ordinal_date(date).ordinal; }
template<class DATE> inline Week get_week(DATE const date) noexcept
  { return safe::get_week_date(date).week; }
template<class DATE> inline Year get_week_year(DATE const date) noexcept
  { return safe::get_week_date(date).week_year; }
template<class DATE> inline Day get_day(DATE const date) noexcept 
  { return safe::get_ymd(date).day;   }

//------------------------------------------------------------------------------
// Arithmetic
//------------------------------------------------------------------------------

/*
 * Returns the date obtained by shifting `date` (signed) `days` forward.
 */
template<class DATE>
inline DATE
days_after(
  DATE const date,
  int const days)
  noexcept
{
  return 
      date.is_valid()
    ? from_offset<DATE>(date.get_offset() + days)
    : DATE::INVALID;
}  


/*
 * Returns the date obtained by shifting `date` (signed) `days` forward.
 */
template<class DATE>
inline DATE
days_before(
  DATE const date,
  int const days)
  noexcept
{
  return safe::days_after(date, -days);
}  


/*
 * Returns the number of days between two dates.
 *
 * If both dates are valid, returns the number of days after `date0` that
 * `date1` occurs; if `date1` is earlier, the result is negative.  If either
 * date is not valid, returns `INT_MIN`.
 */
template<class DATE>
inline int
days_between(
  DATE const date0,
  DATE const date1)
  noexcept
{
  return
      date0.is_valid() && date1.is_valid()
    ? (int) date1.get_offset() - date0.get_offset()
    : std::numeric_limits<int>::min();
}


//------------------------------------------------------------------------------
// Comparisons
//------------------------------------------------------------------------------

template<class DATE>
inline bool
is(
  DATE const date0,
  DATE const date1)
{
  return date0.is(date1);
}


//------------------------------------------------------------------------------

}  // namespace safe
}  // namespace date
}  // namespace cron

