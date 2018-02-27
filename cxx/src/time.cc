#include "ora/time_type.hh"

namespace ora {
namespace time {

//------------------------------------------------------------------------------

template class TimeType<TimeTraits>;
template class TimeType<HiTimeTraits>;
template class TimeType<SmallTimeTraits>;
template class TimeType<NsTimeTraits>;
template class TimeType<Unix32TimeTraits>;
template class TimeType<Unix64TimeTraits>;
template class TimeType<Time128Traits>;

//------------------------------------------------------------------------------

}  // namespace time
}  // namespace ora

