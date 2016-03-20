#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include <cassert>

#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>
#include <numpy/ufuncobject.h>
#include <numpy/npy_3kcompat.h>

#include "py.hh"
#include "PyDate.hh"

using namespace alxs;
using namespace py;

//------------------------------------------------------------------------------

PyArray_Descr*
date_descr;

PyArray_ArrFuncs
date_arrfuncs;


PyObject*
date_getitem(
  void* const data,
  void* const /* arr */)
{
  using Date = PyDateDefault::Date;

  // FIXME: Check PyArray_ISBEHAVED_RO(arr)?
  return PyDateDefault::create(*reinterpret_cast<Date const*>(data));
}


int
date_setitem(
  Object* const item,
  void* const data,
  void* const /* arr */)
{
  using Date = PyDateDefault::Date;

  Date date;
  try {
    date = convert_to_date<Date>(item);
  }
  catch (Exception) {
    return -1;
  }
  *reinterpret_cast<Date*>(data) = date;
  return 0;
}


void
date_copyswap(
  void* const dest,
  void* const src,
  int const swap,
  void* const /* arr */)
{
  using Date = PyDateDefault::Date;

  assert(!swap);  // FIXME
  *reinterpret_cast<Date*>(dest) = *reinterpret_cast<Date const*>(src);
}


void
init_date_dtype(
  Module* const module)
{
  if (_import_array() < 0) 
    throw py::ImportError("failed to import numpy.core.multiarray"); 
  if (_import_umath() < 0) 
    throw py::ImportError("failed to import numpy.core.umath");

  PyArray_InitArrFuncs(&date_arrfuncs);
  date_arrfuncs.getitem     = (PyArray_GetItemFunc*) date_getitem;
  date_arrfuncs.setitem     = (PyArray_SetItemFunc*) date_setitem;
  date_arrfuncs.copyswap    = (PyArray_CopySwapFunc*) date_copyswap;

  date_descr = PyObject_New(PyArray_Descr, &PyArrayDescr_Type);
  Py_INCREF(&PyDateDefault::type_);
  date_descr->typeobj       = &PyDateDefault::type_;
  date_descr->kind          = 'V';
  date_descr->type          = 'j';
  date_descr->byteorder     = '=';
  date_descr->type_num      = 0; /* assigned at registration */
  date_descr->elsize        = sizeof(PyDateDefault::Date);  // FIXME
  date_descr->alignment     = alignof(PyDateDefault::Date);
  date_descr->subarray      = NULL;
  date_descr->fields        = NULL;
  date_descr->names         = NULL;
  date_descr->f             = &date_arrfuncs;

  auto const num = PyArray_RegisterDataType(date_descr);
  if (num < 0)
    throw py::Exception();

  module->AddObject("dtype", (PyObject*) date_descr);
}


