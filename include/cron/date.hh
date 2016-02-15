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
ordinal_is_valid(
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
  return jan1_datenum(year) + ordinal;
}


extern OrdinalDateParts datenum_to_ordinal_date_parts(Datenum);
extern DateParts datenum_to_parts(Datenum, OrdinalDateParts const&);
extern WeekDateParts datenum_to_week_date_parts(Datenum, OrdinalDateParts const&, DateParts const&);

extern inline DateParts
datenum_to_parts(
  Datenum const datenum)
{
  return datenum_to_parts(datenum, datenum_to_ordinal_date_parts(datenum));
}


extern inline WeekDateParts
datenum_to_week_date_parts(
  Datenum const datenum)
{
  return datenum_to_week_date_parts(
    datenum,
    datenum_to_ordinal_date_parts(datenum),
    datenum_to_parts(datenum));
}


//------------------------------------------------------------------------------
// Generic date type.
//------------------------------------------------------------------------------

/**
 * This class represents a date as a datenum stored as an offset of type
 * `TRAITS::Offset` plus a fixed `TRAITS::base`.  
 */
template<class TRAITS>
class DateTemplate
{
public:

  typedef typename TRAITS::Offset Offset;

  // These are declared const here but defined constexpr to work around a clang bug.
  // http://stackoverflow.com/questions/11928089/static-constexpr-member-of-same-type-as-class-being-defined
  static DateTemplate const MIN;
  static DateTemplate const LAST;
  static DateTemplate const MAX;
  static DateTemplate const INVALID;
  static DateTemplate const MISSING;
  static bool constexpr USE_INVALID = TRAITS::use_invalid;

  // Constructors.

  constexpr DateTemplate()
    : offset_(TRAITS::use_invalid ? TRAITS::invalid : TRAITS::min)
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
          USE_INVALID && date.is_invalid() ? INVALID.get_offset()
        : USE_INVALID && date.is_missing() ? MISSING.get_offset()
        : datenum_to_offset(date.get_datenum()))
  {
  }

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
        USE_INVALID && date.is_invalid() ? INVALID.get_offset()
      : USE_INVALID && date.is_missing() ? MISSING.get_offset()
      : datenum_to_offset(date.get_datenum());
    return *this;
  }

  // Factory methods.  

  static DateTemplate 
  from_datenum(
    Datenum datenum) 
  { 
    return DateTemplate(datenum_to_offset(datenum)); 
  }

  static constexpr DateTemplate 
  from_offset(
    Offset offset) 
  { 
    return DateTemplate(validate_offset(offset)); 
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
    return DateTemplate(ymdi / 10000, (ymdi % 10000) / 100 - 1, ymdi % 100 - 1);
  }

  // Accessors.

  bool      is_valid()      const { return in_interval(MIN.offset_, offset_, MAX.offset_); }
  bool      is_invalid()    const { return offset_ == TRAITS::invalid; }
  bool      is_missing()    const { return offset_ == TRAITS::missing; }

  Offset    get_offset()    const { return offset_; }
  Datenum   get_datenum()   const { return is_valid() ? (TRAITS::base + offset_) : DATENUM_INVALID; }
  DateParts get_parts()     const { return datenum_to_parts(get_datenum()); }
  Weekday   get_weekday()   const { return is_valid() ? alxs::cron::get_weekday(get_datenum()) : WEEKDAY_INVALID; }

  OrdinalDateParts  get_ordinal_date_parts()    const { return datenum_to_ordinal_date_parts(get_datenum()); }
  WeekDateParts     get_week_date_parts()       const { return datenum_to_week_date_parts(get_datenum()); }

  bool is(DateTemplate const& o) const { return offset_ == o.offset_; }
  bool operator==(DateTemplate const& o) const { return is_valid() && o.is_valid() && offset_ == o.offset_; }
  bool operator!=(DateTemplate const& o) const { return is_valid() && o.is_valid() && offset_ != o.offset_; }
  bool operator< (DateTemplate const& o) const { return is_valid() && o.is_valid() && offset_ <  o.offset_; }
  bool operator<=(DateTemplate const& o) const { return is_valid() && o.is_valid() && offset_ <= o.offset_; }
  bool operator> (DateTemplate const& o) const { return is_valid() && o.is_valid() && offset_ >  o.offset_; }
  bool operator>=(DateTemplate const& o) const { return is_valid() && o.is_valid() && offset_ >= o.offset_; }

protected:

  template<class EXC>
  static Offset
  on_error()
  {
    if (TRAITS::use_invalid)
      return INVALID.get_offset();
    else
      throw EXC();
  }

  
  static Offset 
  validate_offset(
    Offset offset)
  {
    return 
      in_interval(TRAITS::min, offset, TRAITS::max) 
      ? offset 
      : on_error<DateRangeError>();
  }

  /**
   * Does not handle DATENUM_INVALID.
   */
  static Offset 
  datenum_to_offset(
    Datenum datenum)
  {
    if (! datenum_is_valid(datenum))
      return on_error<InvalidDateError>();
    Datenum const offset = datenum - (Datenum) TRAITS::base;
    return 
      in_interval<Datenum>(TRAITS::min, offset, TRAITS::max) 
      ? offset
      : on_error<DateRangeError>();
  }

  static Offset
  ordinal_date_to_offset(
    Year year,
    Ordinal ordinal)
  {
    return
      ordinal_is_valid(year, ordinal)
      ? datenum_to_offset(ordinal_date_to_datenum(year, ordinal))
      : on_error<InvalidDateError>();
  }

  static Offset 
  ymd_to_offset(
    Year year, 
    Month month, 
    Day day)
  {
    return 
      ymd_is_valid(year, month, day)
      ? datenum_to_offset(ymd_to_datenum(year, month, day))
      : on_error<InvalidDateError>();
  }

private:

  constexpr DateTemplate(Offset offset) : offset_(offset) {}

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
DateTemplate<TRAITS>::LAST{(Offset) (TRAITS::max - 1)};

template<class TRAITS>
DateTemplate<TRAITS> constexpr
DateTemplate<TRAITS>::MAX{TRAITS::max};

template<class TRAITS>
DateTemplate<TRAITS> constexpr
DateTemplate<TRAITS>::INVALID{TRAITS::invalid};

template<class TRAITS>
DateTemplate<TRAITS> constexpr
DateTemplate<TRAITS>::MISSING{TRAITS::missing};

//------------------------------------------------------------------------------
// Concrete date types
//------------------------------------------------------------------------------

struct DateTraits
{
  typedef uint32_t Offset;

  static Datenum constexpr base     =       0;
  static Offset  constexpr invalid  = 3652060;
  static Offset  constexpr missing  = 3652061;
  static Offset  constexpr min      =       0;   //  0001-01-01.
  static Offset  constexpr max      = 3652059;   // 10000-01-01.
  static bool    constexpr use_invalid = true;
};

typedef DateTemplate<DateTraits> Date;


// FIXME: Use a better name.

struct SafeDateTraits
{
  typedef uint32_t Offset;

  static Datenum constexpr base     =       0;
  static Offset  constexpr invalid  = 3652060;
  static Offset  constexpr missing  = 3652061;
  static Offset  constexpr min      =       0;   //  0001-01-01.
  static Offset  constexpr max      = 3652059;   // 10000-01-01.
  static bool    constexpr use_invalid = false;
};

typedef DateTemplate<SafeDateTraits> SafeDate;

struct SmallDateTraits
{
  typedef uint16_t Offset;

  // FIXME: Would be better for max to be distinct from invalid.
  static Datenum constexpr base     = 719162;
  static Offset  constexpr invalid  = std::numeric_limits<Offset>::max() - 1;
  static Offset  constexpr missing  = invalid + 1;
  static Offset  constexpr min      = 0;         // 1970-01-01.
  static Offset  constexpr max      = invalid;   // 2149-05-05.
  static bool    constexpr use_invalid = true;
};

typedef DateTemplate<SmallDateTraits> SmallDate;

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// Functions.
//------------------------------------------------------------------------------

template<class TRAITS> 
extern inline DateTemplate<TRAITS> 
shift(
  DateTemplate<TRAITS> date, 
  ssize_t shift)
{
  return 
      date.is_invalid() ? DateTemplate<TRAITS>::INVALID
    : date.is_missing() ? DateTemplate<TRAITS>::MISSING
    : DateTemplate<TRAITS>::from_offset(date.get_offset() + shift);
}


template<class TRAITS>
extern inline ssize_t
operator-(
  DateTemplate<TRAITS> date0,
  DateTemplate<TRAITS> date1)
{
  if (date0.is_valid() && date1.is_valid())
    return date0.get_offset() - date1.get_offset();
  else if (DateTemplate<TRAITS>::USE_INVALID)
    // FIXME: What do we do with invalid/missing values?
    return 0;
  else
    throw ValueError("can't subtract invalid dates");
}


template<class TRAITS> DateTemplate<TRAITS> operator+ (DateTemplate<TRAITS> date, ssize_t days) { return shift(date,  days); }
template<class TRAITS> DateTemplate<TRAITS> operator+=(DateTemplate<TRAITS>& date, ssize_t days) { return date = shift(date,  days); }
template<class TRAITS> DateTemplate<TRAITS> operator++(DateTemplate<TRAITS>& date) { return date = shift(date, 1); }
template<class TRAITS> DateTemplate<TRAITS> operator++(DateTemplate<TRAITS>& date, int) { auto old = date; date = shift(date, 1); return old; }
template<class TRAITS> DateTemplate<TRAITS> operator- (DateTemplate<TRAITS> date, ssize_t days) { return shift(date, -days); }
template<class TRAITS> DateTemplate<TRAITS> operator-=(DateTemplate<TRAITS>& date, ssize_t days) { return date = shift(date, -days); }
template<class TRAITS> DateTemplate<TRAITS> operator--(DateTemplate<TRAITS>& date) { return date = shift(date, -1); }
template<class TRAITS> DateTemplate<TRAITS> operator--(DateTemplate<TRAITS>& date, int) { auto old = date; date = shift(date, -1); return old; }

// FIXME: Use DateFormat.
extern DateParts iso_parse(std::string const& text);  

//------------------------------------------------------------------------------

}  // namespace cron
}  // namespace alxs


