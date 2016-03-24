#include <cassert>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>
#include <numpy/ufuncobject.h>
#include <numpy/npy_3kcompat.h>

#include "py.hh"
#include "np_date.hh"

using namespace py;
using namespace alxs;

//------------------------------------------------------------------------------

ref<Object>
set_up_numpy(
  Module* const module,
  Tuple* const args,
  Dict* kw_args)
{
  static char const* const arg_names[] = {nullptr};
  Arg::ParseTupleAndKeywords(args, kw_args, "", arg_names);

  // Import numpy stuff.
  if (_import_array() < 0) 
    throw ImportError("failed to import numpy.core.multiarray"); 
  if (_import_umath() < 0) 
    throw ImportError("failed to import numpy.core.umath");

  DateDtype<PyDate<cron::Date>>::add(module);
  DateDtype<PyDate<cron::SmallDate>>::add(module);

  return none_ref();
}


