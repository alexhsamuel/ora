#include "np.hh"

namespace ora {
namespace py {
namespace np {

//------------------------------------------------------------------------------

char
get_type_char()
{
  static char next = 'n';
  return next++;
}


//------------------------------------------------------------------------------

}  // namespace np
}  // namespace py
}  // namespace ora

