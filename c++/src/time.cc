#include "cron/time_type.hh"

namespace cron {
namespace time {

//------------------------------------------------------------------------------

template class TimeType<TimeTraits>;
template class TimeType<SmallTimeTraits>;
template class TimeType<NsecTimeTraits>;
template class TimeType<Unix32TimeTraits>;
template class TimeType<Unix64TimeTraits>;
template class TimeType<Time128Traits>;

//------------------------------------------------------------------------------

}  // namespace time
}  // namespace cron

