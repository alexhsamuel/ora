/*
 * Template date class.
 */

#pragma once

#include <iostream>  // FIXME: Remove.
#include <limits>
#include <string>

#include "aslib/exc.hh"
#include "aslib/math.hh"
#include "cron/date_math.hh"
#include "cron/types.hh"

namespace cron {

using namespace aslib;

//------------------------------------------------------------------------------
// Generic date type
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
 * For example, the <Date16> class, instantiated with the <Date16Traits>
 * traits class, uses an unsigned 16-bit integer to store date offsets from
 * 1970-01-01, with a maximum date of 2149-06-04.
 */
template<class TRAITS>
class DateTemplate
{
public:

  using Traits = TRAITS;
  using Offset = typename TRAITS::Offset;

  // These are declared const here but defined constexpr to work around a clang
  // bug.  http://stackoverflow.com/questions/11928089/static-constexpr-member-of-same-type-as-class-being-defined
  static DateTemplate const MIN;
  static DateTemplate const MAX;
  static DateTemplate const MISSING;
  static DateTemplate const INVALID;

  // Constructors  -------------------------------------------------------------

  /*
   * Default constructor: an invalid date.
   */
  constexpr 
  DateTemplate()
  : offset_(TRAITS::invalid)
  {
  }

  /*
   * Copy constructor.
   */
  DateTemplate(
    DateTemplate const& date)
  : offset_(date.offset_)
  {
  }

  /*
   * Constructs from another date template instance.
   *
   * If `date` is invalid or missing, constructs a corresponding invalid or
   * missing date.  If date is valid but cannot be represented by this date
   * type, throws <DateRangeError>.
   */
  template<class OTHER_TRAITS> 
  DateTemplate(
    DateTemplate<OTHER_TRAITS> const date)
  : DateTemplate(
        date.is_invalid() ? TRAITS::invalid
      : date.is_missing() ? TRAITS::missing
      : valid_offset<DateRangeError>(
          datenum_to_offset(date.get_datenum())))
  {
  }

  /*
   * Constructs from year, month, day components.
   */
  DateTemplate(
    Year const year,
    Month const month,
    Day const day)
  : offset_(valid_offset<DateRangeError>(ymd_to_offset(year, month, day)))
  {
  }

  // Assignment operators  -----------------------------------------------------

  /*
   * Copy assignment.
   *
   * @return
   *   This object.
   */
  DateTemplate
  operator=(
    DateTemplate const date)
  {
    offset_ = date.offset_;
    return *this;
  }

  /*
   * Assigns from another date template instance.
   *
   * If `date` is invalid or missing, constructs a corresponding invalid or
   * missing date.  If date is valid but cannot be represented by this date
   * type, throws <DateRangeError>.
   *
   * @return
   *   This object.
   */
  template<class OTHER_TRAITS>
  DateTemplate
  operator=(
    DateTemplate<OTHER_TRAITS> const date)
  {
    offset_ = 
        date.is_invalid() ? TRAITS::invalid
      : date.is_missing() ? TRAITS::missing
      : valid_offset<DateRangeError>(
          datenum_to_offset(date.get_datenum()));
    return *this;
  }

  // Factory methods  ----------------------------------------------------------

  /*
   * Creates a date object from an offset, which must be valid and in range.
   *
   * Throws <DateRangeError> if the offset is out of range.
   */
  static constexpr DateTemplate 
  from_offset(
    Offset const offset) 
  { 
    return DateTemplate(valid_offset<DateRangeError>(offset));
  }

  /*
   * Creates a date from a datenum.
   *
   * Throws <InvalidDateError> if the datenum is invalid.
   * Throws <DateRangeError> if the datenum is out of range.
   */
  static DateTemplate 
  from_datenum(
    Datenum const datenum) 
  { 
    if (datenum_is_valid(datenum))
      return from_offset(datenum_to_offset(datenum));
    else
      throw InvalidDateError();
  }

  /*
   * Creates a date from an ordinal date.
   *
   * Throws <InvalidDateError> if the ordinal date is invalid.
   * Throws <DateRangeError> if the ordinal date is out of range.
   */
  static DateTemplate 
  from_ordinal_date(
    Year const year, 
    Ordinal const ordinal) 
  { 
    if (ordinal_date_is_valid(year, ordinal))
      return from_datenum(ordinal_date_to_datenum(year, ordinal));
    else
      throw InvalidDateError();
  }

  /*
   * Creates a date from year, month, and day.
   *
   * Throws <InvalidDateError> if the year, month, and day are invalid.
   * Throws <DateRangeError> if the date is out of range.
   */
  static DateTemplate
  from_ymd(
    Year const year, 
    Month const month, 
    Day const day) 
  {
    return from_offset(ymd_to_offset(year, month, day));
  }

  static DateTemplate
  from_ymd(
    DateParts const& parts) 
  {
    return from_ymd(parts.year, parts.month, parts.day);
  }

  /*
   * Creates a date from a week date.
   *
   * Throws <InvalidDateError> if the week date is invalid.
   * Throws <DateRangeError> if the week date is out of range.
   */
  static DateTemplate
  from_week_date(
    Year const week_year,
    Week const week,
    Weekday const weekday)
  {
    if (week_date_is_valid(week_year, week, weekday))
      return from_datenum(week_date_to_datenum(week_year, week, weekday));
    else
      throw InvalidDateError();
  }

  /*
   * Creates a date from a YMDI.
   *
   * Throws <InvalidDateError> if the YMDI is invalid.
   * Throws <DateRangeError> if the YMDI is out of range.
   */
  static DateTemplate 
  from_ymdi(
    int const ymdi) 
  { 
    if (ymdi_is_valid(ymdi))
      return from_datenum(ymdi_to_datenum(ymdi));
    else
      throw InvalidDateError();
  }

  // Accessors  ----------------------------------------------------------------

  bool      is_valid()      const { return offset_is_valid(offset_); }
  bool      is_invalid()    const { return offset_ == TRAITS::invalid; }
  bool      is_missing()    const { return offset_ == TRAITS::missing; }

  Offset get_offset() const 
    { return valid_offset(); }
  Datenum get_datenum() const 
    { return offset_to_datenum(valid_offset()); }
  OrdinalDate get_ordinal_date() const 
    { return cron::datenum_to_ordinal_date(get_datenum()); }
  YmdDate get_ymd() const
    { return cron::datenum_to_ymd(get_datenum()); }
  Weekday get_weekday() const 
    { return cron::get_weekday(get_datenum()); }
  WeekDate get_week_date() const
    { return cron::datenum_to_week_date(get_datenum()); }
  int get_ymdi() const
    { return cron::datenum_to_ymdi(get_datenum()); }

  // FIXME: Remove this.
  DateParts get_parts()     const { return datenum_to_parts(get_datenum()); }

  // Comparisons  --------------------------------------------------------------

  bool is(DateTemplate const& o) const { return offset_ == o.offset_; }
  bool operator==(DateTemplate const& o) const { return is_valid() && o.is_valid() && offset_ == o.offset_; }
  bool operator!=(DateTemplate const& o) const { return is_valid() && o.is_valid() && offset_ != o.offset_; }
  bool operator< (DateTemplate const& o) const { return is_valid() && o.is_valid() && offset_ <  o.offset_; }
  bool operator<=(DateTemplate const& o) const { return is_valid() && o.is_valid() && offset_ <= o.offset_; }
  bool operator> (DateTemplate const& o) const { return is_valid() && o.is_valid() && offset_ >  o.offset_; }
  bool operator>=(DateTemplate const& o) const { return is_valid() && o.is_valid() && offset_ >= o.offset_; }

public:

  // Helper methods  -----------------------------------------------------------

  static Offset
  datenum_to_offset(
    Datenum const datenum)
  {
    return (Offset) ((long) datenum - (long) TRAITS::base);
  }

  static bool
  offset_is_valid(
    Offset const offset)
  {
    return in_range(TRAITS::min, offset, TRAITS::max);
  }

  static Datenum
  offset_to_datenum(
    Offset const offset)
  {
    return (Datenum) ((int64_t) TRAITS::base + (int64_t) offset);
  }

  /*
   * Returns `offset` if it is valid; else throws EXCEPTION.
   */
  template<class EXCEPTION>
  static Offset
  valid_offset(
    Offset const offset)
  {
    if (offset_is_valid(offset))
      return offset;
    else
      throw EXCEPTION();
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

  // State  --------------------------------------------------------------------

  constexpr 
  DateTemplate(
    Offset offset) 
  : offset_(offset) 
  {
  }

  /*
   * Internal accessor for offset that throws if not valid.
   */
  Offset valid_offset() const 
    { return valid_offset<InvalidDateError>(offset_); }

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

extern template class DateTemplate<DateTraits>;
using Date = DateTemplate<DateTraits>;


struct Date16Traits
{
  using Offset = uint16_t;

  static Datenum constexpr base     = 719162;
  static Offset  constexpr min      = 0;         // 1970-01-01.
  static Offset  constexpr max      = std::numeric_limits<Offset>::max() - 2;
                                                 // 2149-06-04.
  static Offset  constexpr missing  = std::numeric_limits<Offset>::max() - 1;
  static Offset  constexpr invalid  = std::numeric_limits<Offset>::max();
};

extern template class DateTemplate<Date16Traits>;
using Date16 = DateTemplate<Date16Traits>;


//------------------------------------------------------------------------------
// Functions.
//------------------------------------------------------------------------------

namespace {

template<class TRAITS>
void ensure_valid(
  DateTemplate<TRAITS> const date)
{
  if (!date.is_valid())
    throw InvalidDateError();
}


}  // anonymous namespace


template<class TRAITS> 
extern inline DateTemplate<TRAITS> 
operator+(
  DateTemplate<TRAITS> date, 
  int shift)
{
  ensure_valid(date);
  return DateTemplate<TRAITS>::from_offset(date.get_offset() + shift);
}


template<class TRAITS> 
extern inline DateTemplate<TRAITS> 
operator-(
  DateTemplate<TRAITS> date, 
  int shift)
{
  ensure_valid(date);
  return DateTemplate<TRAITS>::from_offset(date.get_offset() - shift);
}


template<class TRAITS>
extern inline int
operator-(
  DateTemplate<TRAITS> date0,
  DateTemplate<TRAITS> date1)
{
  ensure_valid(date0);
  ensure_valid(date1);
  return (int) date0.get_offset() - date1.get_offset();
}


template<class TRAITS> DateTemplate<TRAITS> operator+=(DateTemplate<TRAITS>& date, ssize_t days) { return date = date + days; }
template<class TRAITS> DateTemplate<TRAITS> operator++(DateTemplate<TRAITS>& date) { return date = date + 1; }
template<class TRAITS> DateTemplate<TRAITS> operator++(DateTemplate<TRAITS>& date, int) { auto old = date; date = date + 1; return old; }
template<class TRAITS> DateTemplate<TRAITS> operator-=(DateTemplate<TRAITS>& date, ssize_t days) { return date = date -days; }
template<class TRAITS> DateTemplate<TRAITS> operator--(DateTemplate<TRAITS>& date) { return date = date - 1; }
template<class TRAITS> DateTemplate<TRAITS> operator--(DateTemplate<TRAITS>& date, int) { auto old = date; date = date  -1; return old; }

//------------------------------------------------------------------------------

}  // namespace cron


