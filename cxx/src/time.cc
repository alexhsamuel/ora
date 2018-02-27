#include "ora/time_type.hh"

namespace ora {
namespace time {

//------------------------------------------------------------------------------

template class TimeType<TimeTraits>;
template class TimeType<SmallTimeTraits>;
template class TimeType<NsTimeTraits>;
template class TimeType<Time32Traits>;
template class TimeType<Time64Traits>;
template class TimeType<Time128Traits>;

//------------------------------------------------------------------------------

}  // namespace time
}  // namespace ora

