#pragma once

#include <Python.h>

#include "py.hh"

namespace ora {
namespace py {

//------------------------------------------------------------------------------

class PyDateFmt
  : public ExtensionType
{
public:

  static Type type_;
  static void add_to(Module& module);

  PyDateFmt()
  {
  }

};


//------------------------------------------------------------------------------

}  // namespace py
}  // namespace ora

