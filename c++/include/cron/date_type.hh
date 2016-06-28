/*
 * Template date class.
 */

#pragma once

#include <cstddef>
#include <type_traits>

#include "cron/date_math.hh"
#include "cron/exceptions.hh"
#include "cron/types.hh"

namespace cron {
namespace date {

//------------------------------------------------------------------------------
// Forward declarations
//------------------------------------------------------------------------------

namespace safe {

template<class DATE> DATE from_datenum(Datenum) noexcept;
template<class DATE> DATE from_offset(typename DATE::Offset) noexcept;
template<class DATE> bool equal(DATE, DATE) noexcept;

}  // namespace safe

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

  // Constants -----------------------------------------------------------------

  // These are declared const here but defined constexpr to work around a clang
  // bug.  http://stackoverflow.com/questions/11928089/static-constexpr-member-of-same-type-as-class-being-defined
  static DateTemplate const MIN;
  static DateTemplate const MAX;
  static DateTemplate const MISSING;
  static DateTemplate const INVALID;

  // Constructors  -------------------------------------------------------------

  // FIXME: Using '= default' causes instantiation problems?
  constexpr DateTemplate() noexcept {}

  constexpr DateTemplate(DateTemplate const&) noexcept = default;

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
      : from_datenum(date.get_datenum()))
  {
  }

  ~DateTemplate() noexcept = default;

  // Factory methods  ----------------------------------------------------------

  /*
   * Creates a date from a datenum.
   *
   * Throws <InvalidDateError> if the datenum is not valid.
   * Throws <DateRangeError> if the date cannot be represented with this type.
   */
  static DateTemplate
  from_datenum(
    Datenum const datenum)
  {
    if (datenum_is_valid(datenum)) {
      auto const offset = (int64_t) datenum - (int64_t) Traits::base;
      if (in_range<int64_t>(Traits::min, offset, Traits::max))
        return {(Offset) offset};
      else
        throw DateRangeError();
    }
    else
      throw InvalidDateError();
  }

  /*
   * Creates a date from an offset.
   *
   * Throws <DateRangeError> if the offset is not valid.
   */
  static DateTemplate
  from_offset(
    Offset const offset)
  {
    if (in_range(Traits::min, offset, Traits::max))
      return DateTemplate(offset);
    else
      throw DateRangeError();
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
    noexcept
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
    return *this = 
        date.is_invalid() ? INVALID
      : date.is_missing() ? MISSING
      : from_datenum(date.get_datenum());
  }

  // Accessors  ----------------------------------------------------------------

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

  bool is_invalid() const noexcept { return offset_ == Traits::invalid; }
  bool is_missing() const noexcept { return offset_ == Traits::missing; }

  bool 
  is_valid()
    const noexcept
  {
    return in_range(Traits::min, offset_, Traits::max);
  }

private:

  // State  --------------------------------------------------------------------

  constexpr 
  DateTemplate(
    Offset offset) 
  : offset_(offset)
  {
  }

  Offset offset_ = Traits::invalid;

  template<class DATE> friend DATE cron::date::safe::from_datenum(Datenum) noexcept;
  template<class DATE> friend DATE cron::date::safe::from_offset(typename DATE::Offset) noexcept;
  template<class DATE> friend bool cron::date::safe::equal(DATE, DATE) noexcept;

public:

  /*
   * Returns true iff the memory layout is exactly the offset.
   */
  static bool constexpr 
  is_basic_layout()
  {
    return 
         sizeof(DateTemplate) == sizeof(Offset)
      && offsetof(DateTemplate, offset_) == 0;
  }

};


/*
 * If `date` is invalid, throws `InvalidDateError`.
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

