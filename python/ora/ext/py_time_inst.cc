#include "py.hh"
#include "py_time.hh"
#include "np/np_time.hh"

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
}


//------------------------------------------------------------------------------

}  // namespace ora
}  // namespace py

