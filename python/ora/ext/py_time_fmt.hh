#pragma once

#include <Python.h>

#include "py.hh"
#include "text.hh"

namespace ora {
namespace py {

//------------------------------------------------------------------------------

class PyTimeFmt
  : public ExtensionType
{
public:

  static Type type_;
  static void add_to(Module& module);

  PyTimeFmt(
    int     const  precision=-1,
    string  const& invalid  ="INVALID",
    string  const& missing  ="MISSING")
  : precision_(precision),
    invalid_(palide(invalid, get_width(), "", " ", 1, PAD_POS_LEFT_JUSTIFY)),
    missing_(palide(missing, get_width(), "", " ", 1, PAD_POS_LEFT_JUSTIFY))
  {
  }

  long          get_width() const { return 26 + precision_; }

  int const     precision_;
  string const  invalid_;
  string const  missing_;

};


//------------------------------------------------------------------------------

}  // namespace py
}  // namespace ora

