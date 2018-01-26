#include "py.hh"
#include "PyLocalTime.hh"

using std::string;
using namespace std::string_literals;

namespace ora {
namespace py {

//------------------------------------------------------------------------------

void
PyLocalTime::add_to(
  Module& module,
  string const& name)
{
  // Construct the type struct.
  type_ = build_type(string{module.GetName()} + "." + name);
  // Hand it to Python.
  type_.Ready();

  // Add the type to the module.
  module.add(&type_);
}


//------------------------------------------------------------------------------
// Data members
//------------------------------------------------------------------------------

Type
PyLocalTime::type_;

//------------------------------------------------------------------------------
// Standard type methods
//------------------------------------------------------------------------------

void
PyLocalTime::tp_dealloc(
  PyLocalTime* const self)
{
  self->ob_type->tp_free(self);
}


ref<Unicode>
PyLocalTime::tp_repr(
  PyLocalTime* const self)
{
  return Unicode::from(
    "LocalTime("s
    + self->date_->Repr()->as_utf8_string() 
    + ", " 
    + self->daytime_->Repr()->as_utf8_string() 
    + ")");
}


ref<Unicode>
PyLocalTime::tp_str(
  PyLocalTime* const self)
{
  return Unicode::from(
    self->date_->Str()->as_utf8_string() 
    + " " 
    + self->daytime_->Str()->as_utf8_string());
}


void
PyLocalTime::tp_init(
  PyLocalTime* const self,
  Tuple* const args,
  Dict* const kw_args)
{
  // FIXME: Accept other things too.
  Object* date;
  Object* daytime;
  Arg::ParseTuple(args, "OO", &date, &daytime);

  new(self) PyLocalTime(date, daytime);
}


//------------------------------------------------------------------------------
// Getsets
//------------------------------------------------------------------------------

ref<Object>
PyLocalTime::get_date(
  PyLocalTime* const self,
  void* /* closure */)
{
  return self->date_.inc();
}


ref<Object>
PyLocalTime::get_daytime(
  PyLocalTime* const self,
  void* /* closure */)
{
  return self->daytime_.inc();
}


GetSets<PyLocalTime>
PyLocalTime::tp_getsets_
  = GetSets<PyLocalTime>()
    .template add_get<get_date>             ("date")
    .template add_get<get_daytime>          ("daytime")
  ;


//------------------------------------------------------------------------------
// Type object
//------------------------------------------------------------------------------

Type
PyLocalTime::build_type(
  string const& type_name)
{
  return PyTypeObject{
    PyVarObject_HEAD_INIT(nullptr, 0)
    (char const*)         strdup(type_name.c_str()),      // tp_name
    (Py_ssize_t)          sizeof(PyLocalTime),            // tp_basicsize
    (Py_ssize_t)          0,                              // tp_itemsize
    (destructor)          wrap<PyLocalTime, tp_dealloc>,  // tp_dealloc
    (printfunc)           nullptr,                        // tp_print
    (getattrfunc)         nullptr,                        // tp_getattr
    (setattrfunc)         nullptr,                        // tp_setattr
                          nullptr,                        // tp_reserved
    (reprfunc)            wrap<PyLocalTime, tp_repr>,     // tp_repr
    (PyNumberMethods*)    nullptr,                        // tp_as_number
    (PySequenceMethods*)  nullptr,                        // tp_as_sequence
    (PyMappingMethods*)   nullptr,                        // tp_as_mapping
    (hashfunc)            nullptr,                        // tp_hash
    (ternaryfunc)         nullptr,                        // tp_call
    (reprfunc)            wrap<PyLocalTime, tp_str>,      // tp_str
    (getattrofunc)        nullptr,                        // tp_getattro
    (setattrofunc)        nullptr,                        // tp_setattro
    (PyBufferProcs*)      nullptr,                        // tp_as_buffer
    (unsigned long)       Py_TPFLAGS_DEFAULT
                          | Py_TPFLAGS_BASETYPE,          // tp_flags
    (char const*)         nullptr,                        // tp_doc
    (traverseproc)        nullptr,                        // tp_traverse
    (inquiry)             nullptr,                        // tp_clear
    (richcmpfunc)         nullptr,                        // tp_richcompare
    (Py_ssize_t)          0,                              // tp_weaklistoffset
    (getiterfunc)         nullptr,                        // tp_iter
    (iternextfunc)        nullptr,                        // tp_iternext
    (PyMethodDef*)        nullptr,                        // tp_methods
    (PyMemberDef*)        nullptr,                        // tp_members
    (PyGetSetDef*)        tp_getsets_,                    // tp_getset
    (_typeobject*)        nullptr,                        // tp_base
    (PyObject*)           nullptr,                        // tp_dict
    (descrgetfunc)        nullptr,                        // tp_descr_get
    (descrsetfunc)        nullptr,                        // tp_descr_set
    (Py_ssize_t)          0,                              // tp_dictoffset
    (initproc)            wrap<PyLocalTime, tp_init>,     // tp_init
    (allocfunc)           nullptr,                        // tp_alloc
    (newfunc)             PyType_GenericNew,              // tp_new
    (freefunc)            nullptr,                        // tp_free
    (inquiry)             nullptr,                        // tp_is_gc
    (PyObject*)           nullptr,                        // tp_bases
    (PyObject*)           nullptr,                        // tp_mro
    (PyObject*)           nullptr,                        // tp_cache
    (PyObject*)           nullptr,                        // tp_subclasses
    (PyObject*)           nullptr,                        // tp_weaklist
    (destructor)          nullptr,                        // tp_del
    (unsigned int)        0,                              // tp_version_tag
    (destructor)          nullptr,                        // tp_finalize
  };
}


//------------------------------------------------------------------------------

}  // namespace py
}  // namespace ora

