#pragma once

namespace aslib {
namespace docstring {

using doc_t = char const* const;

//------------------------------------------------------------------------------

namespace pydaytime {

extern doc_t type;
extern doc_t from_daytick;
extern doc_t from_hms;
extern doc_t from_ssm;
extern doc_t daytick;
extern doc_t hour;

}  // namespace pydaytime

//------------------------------------------------------------------------------

}  // namespace docstring
}  // namespace aslib

