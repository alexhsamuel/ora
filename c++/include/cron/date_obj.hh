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
        date.is_invalid() ? INVALID
      : date.is_missing() ? MISSING
      : from_datenum<DateTemplate>(date.get_datenum()))
  {
  }

  /*
   * Constructs from year, month, day components.
   */
  DateTemplate(
    Year const year,
    Month const month,
    Day const day)
  : DateTemplate(from_ymd<DateTemplate>(year, month, day))
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
    *this = 
        date.is_invalid() ? INVALID
      : date.is_missing() ? MISSING
      : from_datenum<DateTemplate>(date.get_datenum());
    return *this;
  }

  // Accessors  ----------------------------------------------------------------

  bool 
  is_valid()
    const noexcept
  {
    return in_range(TRAITS::min, offset_, TRAITS::max);
  }

  bool is_invalid() const noexcept { return offset_ == TRAITS::invalid; }
  bool is_missing() const noexcept { return offset_ == TRAITS::missing; }

  Datenum 
  get_datenum() 
    const 
  {
    ensure_valid(*this);
    return (Datenum) ((long) TRAITS::base + (long) offset_);
  }

  Offset 
  get_offset() 
    const 
  { 
    ensure_valid(*this);
    return offset_;
  }

  // Comparisons  --------------------------------------------------------------

  bool is(DateTemplate const& o) const { return offset_ == o.offset_; }
  bool operator==(DateTemplate const& o) const { return is_valid() && o.is_valid() && offset_ == o.offset_; }
  bool operator!=(DateTemplate const& o) const { return is_valid() && o.is_valid() && offset_ != o.offset_; }
  bool operator< (DateTemplate const& o) const { return is_valid() && o.is_valid() && offset_ <  o.offset_; }
  bool operator<=(DateTemplate const& o) const { return is_valid() && o.is_valid() && offset_ <= o.offset_; }
  bool operator> (DateTemplate const& o) const { return is_valid() && o.is_valid() && offset_ >  o.offset_; }
  bool operator>=(DateTemplate const& o) const { return is_valid() && o.is_valid() && offset_ >= o.offset_; }

private:

  template<class DATE> friend DATE cron::date::from_datenum(Datenum);
  template<class DATE> friend DATE cron::date::from_offset(typename DATE::Offset);
  template<class DATE> friend DATE cron::date::safe::from_datenum(Datenum) noexcept;
  template<class DATE> friend DATE cron::date::safe::from_offset(typename DATE::Offset) noexcept;

  // State  --------------------------------------------------------------------

  constexpr 
  DateTemplate(
    Offset offset) 
  : offset_(offset)
  {
  }

  Offset offset_;

public:

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

};


/*
 * If `date` is valid, does nothing; otherwise, throws `InvalidDateError`.
 */
template<class DATE>
void
ensure_valid(
  DATE const date)
{
  if (!date.is_valid())
    throw InvalidDateError();
}


//------------------------------------------------------------------------------
// Day arithmetic
//------------------------------------------------------------------------------

// FIXME: Template on DATE instead?
// FIXME: Move elsewhere.

template<class TRAITS> 
extern inline DateTemplate<TRAITS> 
operator+(
  DateTemplate<TRAITS> date, 
  int shift)
{
  ensure_valid(date);
  return from_offset<DateTemplate<TRAITS>>(date.get_offset() + shift);
}


template<class TRAITS> 
extern inline DateTemplate<TRAITS> 
operator-(
  DateTemplate<TRAITS> date, 
  int shift)
{
  ensure_valid(date);
  return from_offset<DateTemplate<TRAITS>>(date.get_offset() - shift);
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

