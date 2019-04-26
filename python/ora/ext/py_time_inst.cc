#include "py.hh"
#include "py_time.hh"
#include "np_time.hh"

//------------------------------------------------------------------------------

namespace ora {
namespace py {

namespace {

template<class TIME>
void
add_time(
  char const* name,
  Module* const mod,
  Module* const np_mod)
{
  // If we have numpy, make this type a subtype of numpy.generic.  This is
  // necessary for some numpy operations to work.
  auto const base = np_mod == nullptr ? nullptr : (Type*) &PyGenericArrType_Type;

  Type* type = PyTime<TIME>::set_up("ora."s + name, base);
  mod->AddObject(name, (Object*) type);
  if (np_mod != nullptr)
    TimeDtype<PyTime<TIME>>::set_up(np_mod);
}


}  // anonymous namespace


void
set_up_times(
  Module* const mod, 
  Module* const np_mod)
{
  add_time<ora::time::Time>       ("Time"      , mod, np_mod);
  add_time<ora::time::HiTime>     ("HiTime"    , mod, np_mod);
  add_time<ora::time::SmallTime>  ("SmallTime" , mod, np_mod);
  add_time<ora::time::NsTime>     ("NsTime"    , mod, np_mod);
  add_time<ora::time::Unix32Time> ("Unix32Time", mod, np_mod);
  add_time<ora::time::Unix64Time> ("Unix64Time", mod, np_mod);
  add_time<ora::time::Time128>    ("Time128"   , mod, np_mod);

  // This is unfortunate.

  add_time_cast<ora::time::Time, ora::time::HiTime>();
  add_time_cast<ora::time::Time, ora::time::SmallTime>();
  add_time_cast<ora::time::Time, ora::time::NsTime>();
  add_time_cast<ora::time::Time, ora::time::Unix32Time>();
  add_time_cast<ora::time::Time, ora::time::Unix64Time>();
  add_time_cast<ora::time::Time, ora::time::Time128>();

  add_time_cast<ora::time::HiTime, ora::time::Time>();
  add_time_cast<ora::time::HiTime, ora::time::SmallTime>();
  add_time_cast<ora::time::HiTime, ora::time::NsTime>();
  add_time_cast<ora::time::HiTime, ora::time::Unix32Time>();
  add_time_cast<ora::time::HiTime, ora::time::Unix64Time>();
  add_time_cast<ora::time::HiTime, ora::time::Time128>();

  add_time_cast<ora::time::SmallTime, ora::time::Time>();
  add_time_cast<ora::time::SmallTime, ora::time::HiTime>();
  add_time_cast<ora::time::SmallTime, ora::time::NsTime>();
  add_time_cast<ora::time::SmallTime, ora::time::Unix32Time>();
  add_time_cast<ora::time::SmallTime, ora::time::Unix64Time>();
  add_time_cast<ora::time::SmallTime, ora::time::Time128>();

  add_time_cast<ora::time::NsTime, ora::time::Time>();
  add_time_cast<ora::time::NsTime, ora::time::HiTime>();
  add_time_cast<ora::time::NsTime, ora::time::SmallTime>();
  add_time_cast<ora::time::NsTime, ora::time::Unix32Time>();
  add_time_cast<ora::time::NsTime, ora::time::Unix64Time>();
  add_time_cast<ora::time::NsTime, ora::time::Time128>();

  add_time_cast<ora::time::Unix32Time, ora::time::Time>();
  add_time_cast<ora::time::Unix32Time, ora::time::HiTime>();
  add_time_cast<ora::time::Unix32Time, ora::time::SmallTime>();
  add_time_cast<ora::time::Unix32Time, ora::time::NsTime>();
  add_time_cast<ora::time::Unix32Time, ora::time::Unix64Time>();
  add_time_cast<ora::time::Unix32Time, ora::time::Time128>();

  add_time_cast<ora::time::Unix64Time, ora::time::Time>();
  add_time_cast<ora::time::Unix64Time, ora::time::HiTime>();
  add_time_cast<ora::time::Unix64Time, ora::time::SmallTime>();
  add_time_cast<ora::time::Unix64Time, ora::time::NsTime>();
  add_time_cast<ora::time::Unix64Time, ora::time::Unix32Time>();
  add_time_cast<ora::time::Unix64Time, ora::time::Time128>();

  add_time_cast<ora::time::Time128, ora::time::Time>();
  add_time_cast<ora::time::Time128, ora::time::HiTime>();
  add_time_cast<ora::time::Time128, ora::time::SmallTime>();
  add_time_cast<ora::time::Time128, ora::time::NsTime>();
  add_time_cast<ora::time::Time128, ora::time::Unix32Time>();
  add_time_cast<ora::time::Time128, ora::time::Unix64Time>();
}


//------------------------------------------------------------------------------

}  // namespace ora
}  // namespace py

