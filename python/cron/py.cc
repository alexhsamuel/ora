#include "py.hh"

using namespace py;

//------------------------------------------------------------------------------

ref<Bool> const
Bool::TRUE = ref<Bool>::of(Py_True);

ref<Bool> const
Bool::FALSE = ref<Bool>::of(Py_False);

Tuple::Builder<0> const
Tuple::builder;
