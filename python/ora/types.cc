#include "types.hh"

namespace ora {
namespace py {

//------------------------------------------------------------------------------

namespace docstring {

using doc_t = char const* const;
#include "types.docstrings.cc.inc"

}  // namespace docstring


StructSequenceType*
get_ymd_date_type()
{
  static StructSequenceType type;

  if (type.tp_name == nullptr) {
    // Lazy one-time initialization.
    static PyStructSequence_Field fields[] = {
      {(char*) "year"       , nullptr},
      {(char*) "month"      , nullptr},
      {(char*) "day"        , nullptr},
      {nullptr, nullptr}
    };
    static PyStructSequence_Desc desc{
      (char*) "YmdDate",                                    // name
      (char*) docstring::YmdDate,                           // doc
      fields,                                               // fields
      3                                                     // n_in_sequence
    };

    StructSequenceType::InitType(&type, &desc);
  }

  return &type;
}


StructSequenceType*
get_hms_daytime_type()
{
  static StructSequenceType type;

  if (type.tp_name == nullptr) {
    // Lazy one-time initialization.
    static PyStructSequence_Field fields[] = {
      {(char*) "hour"       , nullptr},
      {(char*) "minute"     , nullptr},
      {(char*) "second"     , nullptr},
      {nullptr, nullptr}
    };
    static PyStructSequence_Desc desc{
      (char*) "HmsDaytime",                                 // name
      nullptr,                                              // doc
      fields,                                               // fields
      3                                                     // n_in_sequence
    };

    StructSequenceType::InitType(&type, &desc);
  }

  return &type;
}


ref<Object>
make_hms_daytime(
  ora::HmsDaytime const hms)
{
  auto hms_obj = get_hms_daytime_type()->New();
  hms_obj->initialize(0, Long::FromLong(hms.hour));
  hms_obj->initialize(1, Long::FromLong(hms.minute));
  hms_obj->initialize(2, Float::FromDouble(hms.second));
  return std::move(hms_obj);
}


//------------------------------------------------------------------------------

}  // namespace py
}  // namespace ora

