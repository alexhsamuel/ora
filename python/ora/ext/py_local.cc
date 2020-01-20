#include "py.hh"
#include "py_local.hh"
#include "py_date.hh"
#include "py_daytime.hh"
#include "py_time.hh"
#include "py_time_zone.hh"

using std::string;
using namespace std::string_literals;

namespace ora {
namespace py {

//------------------------------------------------------------------------------

void
PyLocal::add_to(
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
PyLocal::type_;

//------------------------------------------------------------------------------
// Standard type methods
//------------------------------------------------------------------------------

namespace {

void
tp_dealloc(
  PyLocal* const self)
{
  self->ob_type->tp_free(self);
}


ref<Unicode>
tp_repr(
  PyLocal* const self)
{
  return Unicode::from(
    "LocalTime("s
    + self->date_->Repr()->as_utf8_string() 
    + ", " 
    + self->daytime_->Repr()->as_utf8_string() 
    + ")");
}


ref<Unicode>
tp_str(
  PyLocal* const self)
{
  return Unicode::from(
    self->date_->Str()->as_utf8_string() 
    + "T"
    + self->daytime_->Str()->as_utf8_string());
}


ref<Object>
tp_richcompare(
  PyLocal* const self,
  Object* const other,
  int const comparison)
{
  if (Sequence::Check(other)) {
    auto const seq = cast<Sequence>(other);
    if (seq->Length() == 2) {
      auto& d0 = self->date_;
      auto& y0 = self->daytime_;
      auto const d1 = seq->GetItem(0);
      auto const y1 = seq->GetItem(1);

      bool result;
      switch (comparison) {
      case Py_EQ: result = d0->eq(d1) && y0->eq(y1); break;
      case Py_NE: result = d0->ne(d1) || y0->ne(y1); break;
      case Py_LT: result = d0->lt(d1) || (d0->eq(d1) && y0->lt(y1)); break;
      case Py_LE: result = d0->lt(d1) || (d0->eq(d1) && y0->le(y1)); break;
      case Py_GT: result = d0->gt(d1) || (d0->eq(d1) && y0->gt(y1)); break;
      case Py_GE: result = d0->gt(d1) || (d0->eq(d1) && y0->ge(y1)); break;
      default: result = false;  // should be unreachable
      }
      return Bool::from(result);
    }
  }

  return not_implemented_ref();
}


void
tp_init(
  PyLocal* const self,
  Tuple* const args,
  Dict* const kw_args)
{
  Object* date_arg;
  Object* daytime_arg;
  Arg::ParseTuple(args, "OO", &date_arg, &daytime_arg);

  auto const date = to_date_object(date_arg);
  auto const daytime = to_daytime_object(daytime_arg);
  new(self) PyLocal(date, daytime);
}


//------------------------------------------------------------------------------
// Number
//------------------------------------------------------------------------------

ref<Object>
nb_matrix_multiply(
  PyLocal* const self,
  Object* const other,
  bool const right)
{
  // We should be on the LHS of the time zone.
  if (right)
    return not_implemented_ref();

  ora::TimeZone_ptr tz = maybe_time_zone(other);
  if (tz == nullptr) 
    return not_implemented_ref();
  else {
    auto const date_api = PyDateAPI::get(self->date_);
    if (date_api == nullptr)
      // Should not happen.
      return not_implemented_ref();
    auto const daytime_api = PyDaytimeAPI::get(self->daytime_);
    if (daytime_api == nullptr)
      // Should not happen.
      return not_implemented_ref();
    auto const datenum = date_api->get_datenum(self->date_);
    auto const daytick = daytime_api->get_daytick(self->daytime_);
    return PyTimeDefault::create(
      ora::from_local<PyTimeDefault::Time>(datenum, daytick, *tz));
  }
}


PyNumberMethods
tp_as_number = {
  (binaryfunc)  nullptr,                        // nb_add
  (binaryfunc)  nullptr,                        // nb_subtract
  (binaryfunc)  nullptr,                        // nb_multiply
  (binaryfunc)  nullptr,                        // nb_remainder
  (binaryfunc)  nullptr,                        // nb_divmod
  (ternaryfunc) nullptr,                        // nb_power
  (unaryfunc)   nullptr,                        // nb_negative
  (unaryfunc)   nullptr,                        // nb_positive
  (unaryfunc)   nullptr,                        // nb_absolute
  (inquiry)     nullptr,                        // nb_bool
  (unaryfunc)   nullptr,                        // nb_invert
  (binaryfunc)  nullptr,                        // nb_lshift
  (binaryfunc)  nullptr,                        // nb_rshift
  (binaryfunc)  nullptr,                        // nb_and
  (binaryfunc)  nullptr,                        // nb_xor
  (binaryfunc)  nullptr,                        // nb_or
  // Work around a NumPy bug (https://github.com/numpy/numpy/issues/10693) by
  // defining nb_int, nb_float that raise TypeError.
  (unaryfunc)   nullptr,                        // nb_int
  (void*)       nullptr,                        // nb_reserved
  (unaryfunc)   nullptr,                        // nb_float
  (binaryfunc)  nullptr,                        // nb_inplace_add
  (binaryfunc)  nullptr,                        // nb_inplace_subtract
  (binaryfunc)  nullptr,                        // nb_inplace_multiply
  (binaryfunc)  nullptr,                        // nb_inplace_remainder
  (ternaryfunc) nullptr,                        // nb_inplace_power
  (binaryfunc)  nullptr,                        // nb_inplace_lshift
  (binaryfunc)  nullptr,                        // nb_inplace_rshift
  (binaryfunc)  nullptr,                        // nb_inplace_and
  (binaryfunc)  nullptr,                        // nb_inplace_xor
  (binaryfunc)  nullptr,                        // nb_inplace_or
  (binaryfunc)  nullptr,                        // nb_floor_divide
  (binaryfunc)  nullptr,                        // nb_true_divide
  (binaryfunc)  nullptr,                        // nb_inplace_floor_divide
  (binaryfunc)  nullptr,                        // nb_inplace_true_divide
  (unaryfunc)   nullptr,                        // nb_index
  (binaryfunc)  wrap<PyLocal, nb_matrix_multiply>, // nb_matrix_multiply
  (binaryfunc)  nullptr,                        // nb_inplace_matrix_multiply
};

//------------------------------------------------------------------------------
// Sequence
//------------------------------------------------------------------------------

Py_ssize_t
sq_length(
  PyLocal* const self)
{
  return 2;
}


ref<Object>
sq_item(
  PyLocal* const self,
  Py_ssize_t const index)
{
  if (index == 0)
    return self->date_.inc();
  else if (index == 1)
    return self->daytime_.inc();
  else
    throw IndexError("index out of range");
}


PySequenceMethods const 
tp_as_sequence = {
  (lenfunc)         sq_length,                          // sq_length
  (binaryfunc)      nullptr,                            // sq_concat
  (ssizeargfunc)    nullptr,                            // sq_repeat
  (ssizeargfunc)    wrap<PyLocal, sq_item>,             // sq_item
  (void*)           nullptr,                            // was_sq_slice
  (ssizeobjargproc) nullptr,                            // sq_ass_item
  (void*)           nullptr,                            // was_sq_ass_slice
  (objobjproc)      nullptr,                            // sq_contains
  (binaryfunc)      nullptr,                            // sq_inplace_concat
  (ssizeargfunc)    nullptr,                            // sq_inplace_repeat
};


//------------------------------------------------------------------------------
// Getsets
//------------------------------------------------------------------------------

ref<Object>
get_date(
  PyLocal* const self,
  void* /* closure */)
{
  return self->date_.inc();
}


ref<Object>
get_datenum(
  PyLocal* const self,
  void* /* closure */)
{
  return self->date_->GetAttrString("datenum", false);
}


ref<Object>
get_day(
  PyLocal* const self,
  void* /* closure */)
{
  return self->date_->GetAttrString("day", false);
}


ref<Object>
get_daytick(
  PyLocal* const self,
  void* /* closure */)
{
  return self->daytime_->GetAttrString("daytick", false);
}


ref<Object>
get_daytime(
  PyLocal* const self,
  void* /* closure */)
{
  return self->daytime_.inc();
}


ref<Object>
get_hour(
  PyLocal* const self,
  void* /* closure */)
{
  return self->daytime_->GetAttrString("hour", false);
}


ref<Object>
get_minute(
  PyLocal* const self,
  void* /* closure */)
{
  return self->daytime_->GetAttrString("minute", false);
}


ref<Object>
get_month(
  PyLocal* const self,
  void* /* closure */)
{
  return self->date_->GetAttrString("month", false);
}


ref<Object>
get_ordinal(
  PyLocal* const self,
  void* /* closure */)
{
  return self->date_->GetAttrString("ordinal", false);
}


ref<Object>
get_second(
  PyLocal* const self,
  void* /* closure */)
{
  return self->daytime_->GetAttrString("second", false);
}


ref<Object>
get_ssm(
  PyLocal* const self,
  void* /* closure */)
{
  return self->daytime_->GetAttrString("ssm", false);
}


ref<Object>
get_week(
  PyLocal* const self,
  void* /* closure */)
{
  return self->date_->GetAttrString("week", false);
}


ref<Object>
get_week_date(
  PyLocal* const self,
  void* /* closure */)
{
  return self->date_->GetAttrString("week_date", false);
}


ref<Object>
get_week_year(
  PyLocal* const self,
  void* /* closure */)
{
  return self->date_->GetAttrString("week_year", false);
}


ref<Object>
get_weekday(
  PyLocal* const self,
  void* /* closure */)
{
  return self->date_->GetAttrString("weekday", false);
}


ref<Object>
get_year(
  PyLocal* const self,
  void* /* closure */)
{
  return self->date_->GetAttrString("year", false);
}


ref<Object>
get_ymdi(
  PyLocal* const self,
  void* /* closure */)
{
  return self->date_->GetAttrString("ymdi", false);
}


// daytick
// hour
// minute
// second
// ssm

GetSets<PyLocal>
tp_getsets_
  = GetSets<PyLocal>()
    .template add_get<get_date>             ("date")
    .template add_get<get_datenum>          ("datenum")
    .template add_get<get_day>              ("day")
    .template add_get<get_daytick>          ("daytick")
    .template add_get<get_daytime>          ("daytime")
    .template add_get<get_hour>             ("hour")
    .template add_get<get_minute>           ("minute")
    .template add_get<get_month>            ("month")
    .template add_get<get_ordinal>          ("ordinal")
    .template add_get<get_second>           ("second")
    .template add_get<get_ssm>              ("ssm")
    .template add_get<get_week>             ("week")
    .template add_get<get_week_date>        ("week_date")
    .template add_get<get_week_year>        ("week_year")
    .template add_get<get_weekday>          ("weekday")
    .template add_get<get_year>             ("year")
    .template add_get<get_ymdi>             ("ymdi")
  ;


}  // anonymous namespace

//------------------------------------------------------------------------------
// Type object
//------------------------------------------------------------------------------

Type
PyLocal::build_type(
  string const& type_name)
{
  return PyTypeObject{
    PyVarObject_HEAD_INIT(nullptr, 0)
    (char const*)         strdup(type_name.c_str()),      // tp_name
    (Py_ssize_t)          sizeof(PyLocal),                // tp_basicsize
    (Py_ssize_t)          0,                              // tp_itemsize
    (destructor)          wrap<PyLocal, tp_dealloc>,      // tp_dealloc
    (printfunc)           nullptr,                        // tp_print
    (getattrfunc)         nullptr,                        // tp_getattr
    (setattrfunc)         nullptr,                        // tp_setattr
                          nullptr,                        // tp_reserved
    (reprfunc)            wrap<PyLocal, tp_repr>,         // tp_repr
    (PyNumberMethods*)    &tp_as_number,                  // tp_as_number
    (PySequenceMethods*)  &tp_as_sequence,                // tp_as_sequence
    (PyMappingMethods*)   nullptr,                        // tp_as_mapping
    (hashfunc)            nullptr,                        // tp_hash
    (ternaryfunc)         nullptr,                        // tp_call
    (reprfunc)            wrap<PyLocal, tp_str>,          // tp_str
    (getattrofunc)        nullptr,                        // tp_getattro
    (setattrofunc)        nullptr,                        // tp_setattro
    (PyBufferProcs*)      nullptr,                        // tp_as_buffer
    (unsigned long)       Py_TPFLAGS_DEFAULT
                          | Py_TPFLAGS_BASETYPE,          // tp_flags
    (char const*)         nullptr,                        // tp_doc
    (traverseproc)        nullptr,                        // tp_traverse
    (inquiry)             nullptr,                        // tp_clear
    (richcmpfunc)         wrap<PyLocal, tp_richcompare>,  // tp_richcompare
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
    (initproc)            wrap<PyLocal, tp_init>,         // tp_init
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

