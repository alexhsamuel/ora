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
  : precision_(precision),
    bad_result_(get_width(), '#'),  // FIXME
    nat_(palide(nat, get_width(), "", " ", 1, PAD_POS_LEFT_JUSTIFY))
  {
  }

  long get_width() const { return 26 + precision_; }

  int       const precision_;
  string    const bad_result_;
  string    const nat_;

};


//------------------------------------------------------------------------------

}  // namespace py
}  // namespace ora

