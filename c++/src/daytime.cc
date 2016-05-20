#include "cron/daytime.hh"

namespace cron {
namespace daytime {

//------------------------------------------------------------------------------

template class DaytimeTemplate<DaytimeTraits>;
template class DaytimeTemplate<Daytime32Traits>;

//------------------------------------------------------------------------------

}  // namespace daytime
}  // namespace cron

