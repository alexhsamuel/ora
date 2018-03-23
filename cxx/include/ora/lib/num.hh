#pragma once

#include <limits>

#include "math.hh"

namespace ora {
namespace num {

//------------------------------------------------------------------------------

namespace {

// FIXME: std::numeric_limits<uint128_t>::max() appears to be incorrect on gcc?!
// This is a workaround.

template<class T>
T constexpr
max_val() 
{
  return std::numeric_limits<T>::max();
}


template<>
uint128_t constexpr
max_val<uint128_t>()
{
  return make_uint128(0xffffffffffffffff, 0xffffffffffffffff);
}


}  // anonymous namespace


struct
Checked
{
  bool overflow = false;

  operator bool() const { return !overflow; }

  template<class TO, class FROM> 
  inline TO
  convert(
    FROM const from)
  {
    if (!(std::numeric_limits<TO>::min() <= from && from <= max_val<TO>()))
      overflow = true;
    return (TO) from;
  }

};



//------------------------------------------------------------------------------

}  // namespace num
}  // namespace ora

