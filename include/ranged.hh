#pragma once

#include <cassert>

namespace alxs {

//------------------------------------------------------------------------------

class RangedError
  : std::exception
{
};


/**
 * A value of type `VAL` that takes values in a range.  
 *
 * The range is a semi-closed interval [`MIN_`, `MAX_`), or if `CLOSED_` is true
 * the closed interval [`MIN_`, `MAX_`].  Instances with values not in the range
 * may be constructed with unchecked(Value).
 */
template<typename VAL, VAL MIN_, VAL MAX_, bool CLOSED_=false, VAL INV_=std::numeric_limits<VAL>::max()>
class Ranged
{
public:

  using Value = VAL;

  static Value  constexpr MIN       = MIN_;
  static Value  constexpr MAX       = MAX_;
  static Ranged constexpr INVALID   = INV_;
  static bool   constexpr CLOSED    = CLOSED_;

  static bool 
  in_range(
    Value val) 
  { 
    return MIN <= val && (CLOSED ? val <= MAX : val < MAX); 
  }

  /**
   * Constructs an instance without checking that `val` is in the range.
   */
  static Ranged constexpr
  unchecked(
    Value val) 
  { 
    return Ranged(val, false); 
  }

  constexpr
  Ranged()
    : val_(MIN)
  {
  }

  /**
   * Non-checking ctor.
   */
  constexpr 
  Ranged(
    Value val, 
    bool)
    : val_(val)
  {
  }

  /**
   * Throws RangeError if the value is not in the range.
   */
  Ranged(
    Value val)
    : Ranged(check(val), false)
  {
  }

  constexpr
  Ranged(
    Ranged const& other)
    : Ranged(other.val_, false)
  {
  }

  /**
   * Copies from another Ranged without checking the value.
   */
  Ranged
  operator=(
    Ranged const& other)
  {
    val_ = other.val_;
    return *this;
  }

  operator Value() const        { return val_; }

  bool in_range() const         { return in_range(val_); }

  // FIXME
  // Ranged operator+=(Value val)  { return operator=(val_ + val); }
  // Ranged operator-=(Value val)  { return operator=(val_ - val); }
  // Ranged operator*=(Value val)  { return operator=(val_ * val); }
  // Ranged operator/=(Value val)  { return operator=(val_ / val); }
  // Ranged operator%=(Value val)  { return operator=(val_ % val); }
  // Ranged operator++()           { return operator+=(1); }
  // Ranged operator++(int)        { Value const old = val_; operator+=(1); return old; }
  // Ranged operator--()           { return operator-=(1); }
  // Ranged operator--(int)        { Value const old = val_; operator-=(1); return old; }

private:

  // Returns true if `val` is in the range; otherwise throws RangeError.
  static Value
  check(
    Value val)
  {
    if (! in_range(val))
      throw RangedError();
    return val;
  }

  Value val_;

};


// template<typename VAL, VAL MIN_, VAL MAX_, bool CLOSED_, VAL INV_>
// typename Ranged<VAL, MIN_, MAX_, CLOSED_, INV_>::Value constexpr
// Ranged<VAL, MIN_, MAX_, CLOSED_, INV_>::MIN;

// template<typename VAL, VAL MIN_, VAL MAX_, bool CLOSED_, VAL INV_>
// typename Ranged<VAL, MIN_, MAX_, CLOSED_, INV_>::Value constexpr
// Ranged<VAL, MIN_, MAX_, CLOSED_, INV_>::MAX;

// template<typename VAL, VAL MIN_, VAL MAX_, bool CLOSED_, VAL INV_>
// Ranged<VAL, MIN_, MAX_, CLOSED_, INV_> constexpr
// Ranged<VAL, MIN_, MAX_, CLOSED_, INV_>::INVALID;

// template<typename VAL, VAL MIN_, VAL MAX_, bool CLOSED_, VAL INV_>
// bool constexpr
// Ranged<VAL, MIN_, MAX_, CLOSED_, INV_>::CLOSED;

//------------------------------------------------------------------------------

}  // namespace alxs


