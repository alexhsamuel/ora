#pragma once

namespace aslib {
namespace docstring {

using doc_t = char const* const;

//------------------------------------------------------------------------------

namespace pydate {

extern doc_t type;
extern doc_t datenum;
extern doc_t day;
extern doc_t from_datenum;
extern doc_t from_iso_date;
extern doc_t from_offset;
extern doc_t from_ordinal_date;
extern doc_t from_week_date;
extern doc_t from_ymd;
extern doc_t from_ymdi;
extern doc_t invalid;
extern doc_t missing;
extern doc_t month;
extern doc_t offset;
extern doc_t ordinal;
extern doc_t ordinal_date;
extern doc_t valid;
extern doc_t week;
extern doc_t week_date;
extern doc_t week_year;
extern doc_t weekday;
extern doc_t year;
extern doc_t ymdi;
extern doc_t ymd;

}  // namespace pydate

//------------------------------------------------------------------------------

namespace ymddate {

extern doc_t type;

}  // namespace ymddate

//------------------------------------------------------------------------------

}  // namespace docstring
}  // namespace aslib

