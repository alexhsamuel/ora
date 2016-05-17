/*
 * Template date class.
 */

#pragma once

#include <type_traits>

#include "cron/date_math.hh"
#include "cron/types.hh"

namespace cron {
namespace date {

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

  // Accessors  ----------------------------------------------------------------

  bool      is_valid()      const { return offset_is_valid(offset_); }
  bool      is_invalid()    const { return offset_ == TRAITS::invalid; }
  bool      is_missing()    const { return offset_ == TRAITS::missing; }

  Offset get_offset() const 
    { return valid_offset(); }
  Datenum get_datenum() const 
    { return offset_to_datenum(valid_offset()); }

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

  /*
   * Computes the offset for a datenum.
   *
   * Returns the invalid offset if the datenum is outside the range of this
   * template instance.
   */
  static Offset
  datenum_to_offset(
    Datenum const datenum)
  {
    auto offset = (long) datenum - (long) TRAITS::base;
    return overflows<Offset>(offset) ? TRAITS::invalid : (Offset) offset;
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

  // FIXME: Obviate?
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

  /*
   * Returns true iff the memory layout is exactly the offset.
   */
  static constexpr bool 
  is_basic_layout()
  {
    return 
         sizeof(DateTemplate) == sizeof(Offset)
      && offsetof(DateTemplate, offset_) == 0;
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
DateTemplate<TRAITS>::MIN
  {TRAITS::min};

template<class TRAITS>
DateTemplate<TRAITS> constexpr
DateTemplate<TRAITS>::MAX
  {TRAITS::max};

template<class TRAITS>
DateTemplate<TRAITS> constexpr
DateTemplate<TRAITS>::MISSING
  {TRAITS::missing};

template<class TRAITS>
DateTemplate<TRAITS> constexpr
DateTemplate<TRAITS>::INVALID
  {TRAITS::invalid};

//------------------------------------------------------------------------------
// Date template instances
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
static_assert(Date::is_basic_layout(), "wrong memory layout for Date");

//------------------------------------------------------------------------------

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
static_assert(Date16::is_basic_layout(), "wrong memory layout for Date16");

//------------------------------------------------------------------------------

}  // namespace date
}  // namespace cron
