#pragma once

#include "ora.hh"
#include "py.hh"

namespace ora {
namespace py {

//------------------------------------------------------------------------------

extern StructSequenceType* get_ymd_date_type();
extern StructSequenceType* get_hms_daytime_type();
extern ref<Object> make_hms_daytime(ora::HmsDaytime);

//------------------------------------------------------------------------------

}  // namespace py
}  // namespace ora

