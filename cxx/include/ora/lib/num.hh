#pragma once

#include <limits>

namespace ora {
namespace num {

//------------------------------------------------------------------------------

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
    if (!(   from >= std::numeric_limits<TO>::min()
          && from <= std::numeric_limits<TO>::max()))
      overflow = true;
    return (TO) from;
  }

};



//------------------------------------------------------------------------------

}  // namespace num
}  // namespace ora

