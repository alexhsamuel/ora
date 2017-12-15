#include "cron/daytime_type.hh"

namespace cron {
namespace daytime {

//------------------------------------------------------------------------------

template class DaytimeTemplate<DaytimeTraits>;
template class DaytimeTemplate<Daytime32Traits>;
template class DaytimeTemplate<UsecDaytimeTraits>;

//------------------------------------------------------------------------------

}  // namespace daytime
}  // namespace cron

