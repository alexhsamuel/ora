#pragma once

#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include <Python.h>

#include "py.hh"
#include "numpy.hh"

namespace aslib {

using namespace py;

//------------------------------------------------------------------------------

extern PyArray_Descr* get_ymd_dtype();

//------------------------------------------------------------------------------

}  // namespace aslib

