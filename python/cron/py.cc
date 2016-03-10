#include "py.hh"

namespace py {

//------------------------------------------------------------------------------

ref<Object> const
None
  = ref<Object>::take(Py_None);

ref<Bool> const
Bool::TRUE
  = ref<Bool>::of(Py_True);

ref<Bool> const
Bool::FALSE
  = ref<Bool>::of(Py_False);

Tuple::Builder<0> const
Tuple::builder;

//------------------------------------------------------------------------------

}  // namespace py

