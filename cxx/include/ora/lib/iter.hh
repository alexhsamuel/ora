#pragma once

#include <experimental/optional>

namespace ora {
namespace lib {

//------------------------------------------------------------------------------

using std::experimental::optional;

/*
 * Virtual, simple (Python-style) iterator.
 */
template<class T>
class Iter
{
public:

  virtual ~Iter() = default;

  /*
   * Returns the next item in the iterator, if any.
   */
  virtual optional<T> next() = 0;

};


//------------------------------------------------------------------------------

}  // namespace lib
}  // namespace ora

