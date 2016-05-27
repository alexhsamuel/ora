#include "cron/time.hh"

namespace cron {
namespace time {

//------------------------------------------------------------------------------

template class TimeTemplate<TimeTraits>;
template class TimeTemplate<SmallTimeTraits>;
template class TimeTemplate<NsecTimeTraits>;
template class TimeTemplate<Unix32TimeTraits>;
template class TimeTemplate<Unix64TimeTraits>;
template class TimeTemplate<Time128Traits>;

//------------------------------------------------------------------------------

}  // namespace time
}  // namespace cron

