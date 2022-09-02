#include <iostream>

#include <Python.h>

#include "ora/format.hh"
#include "py.hh"
#include "py_time.hh"
#include "py_time_fmt.hh"

using std::string;

namespace ora {
namespace py {

//------------------------------------------------------------------------------

void
PyTimeFmt::add_to(
  Module& module)
{
  type_.Ready();
  module.add(&type_);
}


namespace {

void tp_init(PyTimeFmt* self, Tuple* args, Dict* kw_args)
{
  static char const* arg_names[] = {
    "precision", "invalid", "missing", nullptr };
  Object*       precision_arg   = (Object*) Py_None;
  char const*   invalid         = "INVALID";
  char const*   missing         = "MISSING";
  Arg::ParseTupleAndKeywords(
    args, kw_args, "|O$etet", arg_names,
    &precision_arg, "utf-8", &invalid, "utf-8", &missing);

  auto const precision
    = precision_arg == Py_None ? -1
      : std::max(precision_arg->long_value(), -1l);

  new(self) PyTimeFmt(precision, invalid, missing);
}


ref<Unicode> tp_repr(PyTimeFmt* self)
{
  std::stringstream ss;
  ss << "TimeFmt(" << self->precision_
     << ", invalid=\"" << self->invalid_
     << "\", missing=\"" << self->missing_
     << "\")";
  return Unicode::from(ss.str());
}


ref<Object> tp_call(PyTimeFmt* self, Tuple* args, Dict* kw_args)
{
  static char const* arg_names[] = { "value", nullptr };
  Object* arg;
  Arg::ParseTupleAndKeywords(args, kw_args, "O", arg_names, &arg);

  // Use high resolution time here so we can round correctly.
  auto const time = convert_to_time<NsTime>(arg);
  if (time.is_invalid())
    return Unicode::from(self->invalid_);
  else if (time.is_missing())
    return Unicode::from(self->missing_);
  else {
    // Round to nearest instead of truncate.
    auto const ldd = to_local_datenum_daytick(time + self->round_step_, *UTC);
    StringBuilder sb;
    time::format_iso_time(
      sb, datenum_to_ymd(ldd.datenum), daytick_to_hms(ldd.daytick),
      ldd.time_zone, self->precision_);
    return Unicode::FromStringAndSize((char const*) sb, sb.length());
  }
}


auto methods = Methods<PyTimeFmt>();


ref<Object> get_invalid(PyTimeFmt* const self, void* /* closure */)
{
  return Unicode::from(self->invalid_);
}


ref<Object> get_missing(PyTimeFmt* const self, void* /* closure */)
{
  return Unicode::from(self->missing_);
}


ref<Object> get_precision(PyTimeFmt* const self, void* /* closure */)
{
  return Long::FromLong(self->precision_);
}


ref<Object> get_width(PyTimeFmt* const self, void* /* closure */)
{
  return Long::FromLong(self->get_width());
}


auto getsets = GetSets<PyTimeFmt>()
  .add_get<get_invalid>     ("invalid")
  .add_get<get_missing>     ("missing")
  .add_get<get_precision>   ("precision")
  .add_get<get_width>       ("width")
  ;


}  // anonymous namespace


Type PyTimeFmt::type_ = PyTypeObject{
  PyVarObject_HEAD_INIT(nullptr, 0)
  (char const*)         "TimeFmt",                          // tp_name
  (Py_ssize_t)          sizeof(PyTimeFmt),                  // tp_basicsize
  (Py_ssize_t)          0,                                  // tp_itemsize
  (destructor)          nullptr,                            // tp_dealloc
  (printfunc)           nullptr,                            // tp_print
  (getattrfunc)         nullptr,                            // tp_getattr
  (setattrfunc)         nullptr,                            // tp_setattr
  (PyAsyncMethods*)     nullptr,                            // tp_as_async
  (reprfunc)            wrap<PyTimeFmt, tp_repr>,           // tp_repr
  (PyNumberMethods*)    nullptr,                            // tp_as_number
  (PySequenceMethods*)  nullptr,                            // tp_as_sequence
  (PyMappingMethods*)   nullptr,                            // tp_as_mapping
  (hashfunc)            nullptr,                            // tp_hash
  (ternaryfunc)         wrap<PyTimeFmt, tp_call>,           // tp_call
  (reprfunc)            nullptr,                            // tp_str
  (getattrofunc)        nullptr,                            // tp_getattro
  (setattrofunc)        nullptr,                            // tp_setattro
  (PyBufferProcs*)      nullptr,                            // tp_as_buffer
  (unsigned long)       Py_TPFLAGS_DEFAULT
                        | Py_TPFLAGS_BASETYPE,              // tp_flags
  (char const*)         nullptr,                            // tp_doc
  (traverseproc)        nullptr,                            // tp_traverse
  (inquiry)             nullptr,                            // tp_clear
  (richcmpfunc)         nullptr,                            // tp_richcompare
  (Py_ssize_t)          0,                                  // tp_weaklistoffset
  (getiterfunc)         nullptr,                            // tp_iter
  (iternextfunc)        nullptr,                            // tp_iternext
  (PyMethodDef*)        methods,                            // tp_methods
  (PyMemberDef*)        nullptr,                            // tp_members
  (PyGetSetDef*)        getsets,                            // tp_getset
  (_typeobject*)        nullptr,                            // tp_base
  (PyObject*)           nullptr,                            // tp_dict
  (descrgetfunc)        nullptr,                            // tp_descr_get
  (descrsetfunc)        nullptr,                            // tp_descr_set
  (Py_ssize_t)          0,                                  // tp_dictoffset
  (initproc)            wrap<PyTimeFmt, tp_init>,           // tp_init
  (allocfunc)           nullptr,                            // tp_alloc
  (newfunc)             PyType_GenericNew,                  // tp_new
  (freefunc)            nullptr,                            // tp_free
  (inquiry)             nullptr,                            // tp_is_gc
  (PyObject*)           nullptr,                            // tp_bases
  (PyObject*)           nullptr,                            // tp_mro
  (PyObject*)           nullptr,                            // tp_cache
  (PyObject*)           nullptr,                            // tp_subclasses
  (PyObject*)           nullptr,                            // tp_weaklist
  (destructor)          nullptr,                            // tp_del
  (unsigned int)        0,                                  // tp_version_tag
  (destructor)          nullptr,                            // tp_finalize
};


//------------------------------------------------------------------------------

}  // namespace py
}  // namespace ora

