#include "py.hh"

#include "np_date.hh"

namespace ora {
namespace py {
namespace docstring {
namespace np_date {

#include "np_date.docstrings.cc.inc"

}  // namespace np_date
}  // namespace docstring

//------------------------------------------------------------------------------

std::vector<bool>
DateAPI::kinds_(128, false);

//------------------------------------------------------------------------------

}  // namespace py
}  // namespace ora

