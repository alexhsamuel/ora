#pragma once

#include <iostream>  // FIXME: Remove.
#include <limits>
#include <string>

#include "cron/math.hh"
#include "cron/types.hh"
#include "exc.hh"

namespace alxs {
namespace cron {

//------------------------------------------------------------------------------
// Functions
//------------------------------------------------------------------------------

extern inline bool constexpr
is_leap_year(
  Year year)
{
  return year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
}


extern inline Ordinal constexpr
ordinals_per_year(
  Year year)
{
  return is_leap_year(year) ? 366 : 365;
}


extern inline Day constexpr
days_per_month(
  Year year,
  Month month)
{
  return 
      month ==  3 || month ==  5 || month ==  8 || month == 10 ? 30
    : month == 1 ? (is_leap_year(year) ? 29 : 28)
    : 31;
}


extern inline bool constexpr
ordinal_date_is_valid(
  Year year,
  Ordinal ordinal)
{
  return 
       year_is_valid(year) 
    && in_interval(ORDINAL_MIN, ordinal, ordinals_per_year(year));
}


extern inline bool constexpr
ymd_is_valid(
  Year year,
  Month month,
  Day day)
{
  return 
       month_is_valid(month)
    && year_is_valid(year)
    && in_interval(DAY_MIN, day, days_per_month(year, month));
}


extern inline Weekday constexpr
get_weekday(
  Datenum datenum)
{
  if (datenum_is_valid(datenum)) 
    // 0001-01-01 is a Monday.
    return (MONDAY + datenum) % 7;
  else
    return WEEKDAY_INVALID;
}


/**
 * Returns the datenum for Jan 1 of 'year'.
 */
inline Datenum constexpr
jan1_datenum(
  Year const year)
{
  return
    // An ordinary year has 365 days; count from year 1.
    365 * (year - 1)
    // Add a leap day for multiples of four; century years are not leap years
    // unless also a multiple of 400.  Subtract one from the year, since we
    // are considering Jan 1 and therefore care about previous years only.
    + (year - 1) /   4
    - (year - 1) / 100
    + (year - 1) / 400;
}


namespace impl {

/**
 * Returns the offset of the first day of them month relative to the most recent
 * March 1.  This is the same for ordinary and leap years.
 */
inline Datenum constexpr
get_month_offset(
  Year year, 
  Month month)
{
  // The cumbersome construction is required for constexpr.
  return
      (month == 0) ?    0
    : (month == 1) ?   31
    : (
         (month ==  2) ?  59
       : (month ==  3) ?  90
       : (month ==  4) ? 120
       : (month ==  5) ? 151
       : (month ==  6) ? 181
       : (month ==  7) ? 212
       : (month ==  8) ? 243
       : (month ==  9) ? 273
       : (month == 10) ? 304
       :                 334
      ) + (is_leap_year(year) ? 1 : 0);
}


inline Datenum constexpr 
ymd_to_datenum(
  Year year,
  Month month,
  Day day)
{
  return
      jan1_datenum(year)
    + get_month_offset(year, month)
    + day;
}


}  // namespace impl


inline Datenum constexpr 
week_date_to_datenum(
  Year const week_year,
  Week const week,
  Weekday const weekday)
{
  // FIXME: Validate.
  Datenum const jan1 = jan1_datenum(week_year);
  return 
      jan1                              // Start with Jan 1.
    + (10 - get_weekday(jan1)) % 7 - 3  // Adjust to start on the full week.
    + week * 7                          // Add the week offset.
    + weekday;                          // Add the weekday offset.
}


extern inline Datenum constexpr
ymd_to_datenum(
  Year year,
  Month month,
  Day day)
{
  return
      ymd_is_valid(year, month, day) 
    ? impl::ymd_to_datenum(year, month, day)
    : DATENUM_INVALID;
}


extern inline Datenum constexpr 
ordinal_date_to_datenum(
  Year year,
  Ordinal ordinal)
{
  return
      ordinal_date_is_valid(year, ordinal)
    ? jan1_datenum(year) + ordinal
    : DATENUM_INVALID;
}


extern DateParts datenum_to_parts(Datenum datenum);

//------------------------------------------------------------------------------
// Generic date type.
//------------------------------------------------------------------------------

/*
 * Represents a Gregorian date as an integer day offset from a fixed base date.
 *
 * Each template instance is a non-virtual class with a single integer data
 * member, the offset, and no nontrivial destructor behavior.  The class is
 * designed so that a pointer to the underlying integer type can safely be cast
 * to the date class.
 *
 * A template instance is customized by `TRAITS`, which specifies the following:
 *
 * - The base date, as days counted from 0001-01-01.
 * - The integer type of the offset from the base date.
 * - The min and max valid dates.
 * - A flag indicating whether to use a special `INVALID` value instead of
 *   raising an exception.
 * - Offset values used to represent `INVALID` and `MISSING` date values.
 *
 * For example, the <SmallDate> class, instantiated with the <SmallDateTraits>
 * traits class, uses an unsigned 16-bit integer to store date offsets from
 * 1970-01-01, with a maximum date of 2149-06-04.
 */
template<class TRAITS>
class DateTemplate
{
public:

  using Offset = typename TRAITS::Offset;

  // These are declared const here but defined constexpr to work around a clang
  // bug.  http://stackoverflow.com/questions/11928089/static-constexpr-member-of-same-type-as-class-being-defined
  static DateTemplate const MIN;
  static DateTemplate const MAX;
  static DateTemplate const MISSING;
  static DateTemplate const INVALID;

  // Constructors.

  constexpr 
  DateTemplate()
  : offset_(TRAITS::invalid)
  {
  }

  // FIXME: Should this really be public?
  constexpr 
  DateTemplate(
    Offset offset) 
  : offset_(offset) 
  {
  }

  DateTemplate(
    DateTemplate const& date)
  : offset_(date.offset_)
  {
  }

  template<class OTHER_TRAITS> 
  DateTemplate(
    DateTemplate<OTHER_TRAITS> date)
  : DateTemplate(
        date.is_invalid() ? INVALID.get_offset()
      : date.is_missing() ? MISSING.get_offset()
      : datenum_to_offset(date.get_datenum()))
  {
  }

  // Assignment operators.

  DateTemplate
  operator=(
    DateTemplate const& date)
  {
    offset_ = date.offset_;
    return *this;
  }

  template<class OTHER_TRAITS>
  DateTemplate
  operator=(
    DateTemplate<OTHER_TRAITS> date)
  {
    offset_ = 
        date.is_invalid() ? INVALID.get_offset()
      : date.is_missing() ? MISSING.get_offset()
      : datenum_to_offset(date.get_datenum());
    return *this;
  }

  // Factory methods.  

  static constexpr DateTemplate 
  from_offset(
    Offset const offset) 
  { 
    if (in_range(TRAITS::min, offset, TRAITS::max))
      return DateTemplate(offset); 
    else
      throw DateRangeError();
  }

  static DateTemplate 
  from_datenum(
    Datenum const datenum) 
  { 
    return DateTemplate(datenum_to_offset(datenum)); 
  }

  static DateTemplate 
  from_ordinal_date(
    Year year, 
    Ordinal ordinal) 
  { 
    return DateTemplate(ordinal_date_to_offset(year, ordinal)); 
  }

  static constexpr DateTemplate
  from_parts(
    Year year, 
    Month month, 
    Day day) 
  {
    return DateTemplate(ymd_to_offset(year, month, day));
  }

  static DateTemplate
  from_parts(
    DateParts const& parts) 
  {
    return from_parts(parts.year, parts.month, parts.day);
  }

  static DateTemplate
  from_week_date(
    Year const week_year,
    Week const week,
    Weekday const weekday)
  {
    return DateTemplate(
      datenum_to_offset(week_date_to_datenum(week_year, week, weekday)));
  }

  static DateTemplate 
  from_ymdi(
    int const ymdi) 
  { 
    return DateTemplate::from_parts(ymdi / 10000, (ymdi % 10000) / 100 - 1, ymdi % 100 - 1);
  }

  // Accessors.

  bool      is_valid()      const { return in_range(MIN.offset_, offset_, MAX.offset_); }
  bool      is_invalid()    const { return offset_ == TRAITS::invalid; }
  bool      is_missing()    const { return offset_ == TRAITS::missing; }

  Offset    get_offset()    const { return offset_; }
  Datenum   get_datenum()   const { return is_valid() ? (TRAITS::base + offset_) : DATENUM_INVALID; }
  DateParts get_parts()     const { return datenum_to_parts(get_datenum()); }
  Weekday   get_weekday()   const { return is_valid() ? alxs::cron::get_weekday(get_datenum()) : WEEKDAY_INVALID; }

  bool is(DateTemplate const& o) const { return offset_ == o.offset_; }
  bool operator==(DateTemplate const& o) const { return is_valid() && o.is_valid() && offset_ == o.offset_; }
  bool operator!=(DateTemplate const& o) const { return is_valid() && o.is_valid() && offset_ != o.offset_; }
  bool operator< (DateTemplate const& o) const { return is_valid() && o.is_valid() && offset_ <  o.offset_; }
  bool operator<=(DateTemplate const& o) const { return is_valid() && o.is_valid() && offset_ <= o.offset_; }
  bool operator> (DateTemplate const& o) const { return is_valid() && o.is_valid() && offset_ >  o.offset_; }
  bool operator>=(DateTemplate const& o) const { return is_valid() && o.is_valid() && offset_ >= o.offset_; }

protected:

  /*
   * Does not handle DATENUM_INVALID.
   */
  static Offset 
  datenum_to_offset(
    Datenum const datenum)
  {
    if (! datenum_is_valid(datenum))
      throw InvalidDateError();
    Datenum const offset = datenum - (Datenum) TRAITS::base;
    if (in_range(TRAITS::min, offset, TRAITS::max))
      return offset;
    else
      throw DateRangeError();
  }

  static Offset
  ordinal_date_to_offset(
    Year const year,
    Ordinal const ordinal)
  {
    if (ordinal_is_valid(year, ordinal))
      return datenum_to_offset(ordinal_date_to_datenum(year, ordinal));
    else
      throw InvalidDateError();
  }

  static Offset 
  ymd_to_offset(
    Year const year, 
    Month const month, 
    Day const day)
  {
    if (ymd_is_valid(year, month, day))
      return datenum_to_offset(ymd_to_datenum(year, month, day));
    else
      throw InvalidDateError();
  }

private:

  Offset offset_;

};


//------------------------------------------------------------------------------
// Static attributes
//------------------------------------------------------------------------------

template<class TRAITS>
DateTemplate<TRAITS> constexpr
DateTemplate<TRAITS>::MIN{TRAITS::min};

template<class TRAITS>
DateTemplate<TRAITS> constexpr
DateTemplate<TRAITS>::MAX{TRAITS::max};

template<class TRAITS>
DateTemplate<TRAITS> constexpr
DateTemplate<TRAITS>::MISSING{TRAITS::missing};

template<class TRAITS>
DateTemplate<TRAITS> constexpr
DateTemplate<TRAITS>::INVALID{TRAITS::invalid};

//------------------------------------------------------------------------------
// Concrete date types
//------------------------------------------------------------------------------

struct DateTraits
{
  using Offset = uint32_t;

  static Datenum constexpr base     =       0;
  static Offset  constexpr min      =       0;   // 0001-01-01.
  static Offset  constexpr max      = 3652058;   // 9999-12-31.
  static Offset  constexpr missing  = std::numeric_limits<Offset>::max() - 1;
  static Offset  constexpr invalid  = std::numeric_limits<Offset>::max();
};

using Date = DateTemplate<DateTraits>;

struct SmallDateTraits
{
  using Offset = uint16_t;

  static Datenum constexpr base     = 719162;
  static Offset  constexpr min      = 0;         // 1970-01-01.
  static Offset  constexpr max      = std::numeric_limits<Offset>::max() - 2;
                                                 // 2149-06-04.
  static Offset  constexpr missing  = std::numeric_limits<Offset>::max() - 1;
  static Offset  constexpr invalid  = std::numeric_limits<Offset>::max();
};

using SmallDate = DateTemplate<SmallDateTraits>;

//------------------------------------------------------------------------------
// Functions.
//------------------------------------------------------------------------------

template<class TRAITS> 
extern inline DateTemplate<TRAITS> 
operator+(
  DateTemplate<TRAITS> date, 
  signed int shift)
{
  return 
      date.is_invalid() || date.is_missing() ? date
    : DateTemplate<TRAITS>::from_offset(date.get_offset() + shift);
}


template<class TRAITS> 
extern inline DateTemplate<TRAITS> 
operator-(
  DateTemplate<TRAITS> date, 
  signed int shift)
{
  return 
      date.is_invalid() || date.is_missing() ? date
    : DateTemplate<TRAITS>::from_offset(date.get_offset() - shift);
}


template<class TRAITS>
extern inline ssize_t
operator-(
  DateTemplate<TRAITS> date0,
  DateTemplate<TRAITS> date1)
{
  if (date0.is_valid() && date1.is_valid())
    return (ssize_t) date0.get_offset() - date1.get_offset();
  else
    throw ValueError("can't subtract invalid dates");
}


template<class TRAITS> DateTemplate<TRAITS> operator+=(DateTemplate<TRAITS>& date, ssize_t days) { return date = date + days; }
template<class TRAITS> DateTemplate<TRAITS> operator++(DateTemplate<TRAITS>& date) { return date = date + 1; }
template<class TRAITS> DateTemplate<TRAITS> operator++(DateTemplate<TRAITS>& date, int) { auto old = date; date = date + 1; return old; }
template<class TRAITS> DateTemplate<TRAITS> operator-=(DateTemplate<TRAITS>& date, ssize_t days) { return date = date -days; }
template<class TRAITS> DateTemplate<TRAITS> operator--(DateTemplate<TRAITS>& date) { return date = date - 1; }
template<class TRAITS> DateTemplate<TRAITS> operator--(DateTemplate<TRAITS>& date, int) { auto old = date; date = date  -1; return old; }

// FIXME: Use DateFormat.
extern DateParts iso_parse(std::string const& text);  

//------------------------------------------------------------------------------

}  // namespace cron
}  // namespace alxs


