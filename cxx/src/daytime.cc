#include "ora/daytime_type.hh"

namespace ora {
namespace daytime {

//------------------------------------------------------------------------------

template class DaytimeTemplate<DaytimeTraits>;
template class DaytimeTemplate<Daytime32Traits>;
template class DaytimeTemplate<NsDaytimeTraits>;
template class DaytimeTemplate<UsecDaytimeTraits>;

//------------------------------------------------------------------------------

}  // namespace daytime
}  // namespace ora

