#include "cron/date.hh"

namespace cron {
namespace date {

//------------------------------------------------------------------------------

template class DateTemplate<DateTraits>;
template class DateTemplate<Date16Traits>;

//------------------------------------------------------------------------------

}  // namespace date
}  // namespace cron

