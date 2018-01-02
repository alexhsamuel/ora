#pragma once

namespace ora {
namespace py {
namespace docstring {

using doc_t = char const* const;

//------------------------------------------------------------------------------

extern doc_t type;

extern doc_t days_in_month;
extern doc_t days_in_year;
extern doc_t from_local;
extern doc_t get_display_time_zone;
extern doc_t get_system_time_zone;
extern doc_t get_zoneinfo_dir;
extern doc_t is_leap_year;
extern doc_t now;
extern doc_t set_display_time_zone;
extern doc_t set_zoneinfo_dir;
extern doc_t to_local;
extern doc_t to_local_datenum_daytick;
extern doc_t today;

//------------------------------------------------------------------------------

}  // namespace docstring
}  // namespace py
}  // namespace ora

