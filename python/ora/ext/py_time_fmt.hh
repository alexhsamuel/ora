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
    string  const& nat      ="NaT")
  : precision_(precision)
  {
  }

  long get_width() const { return 26 + precision_; }

  int const precision_;

};


//------------------------------------------------------------------------------

}  // namespace py
}  // namespace ora

