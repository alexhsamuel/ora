#include <cassert>
#include <cstring>
#include <string>

#include "ora/lib/file.hh"
#include "ora/lib/filename.hh"
#include "ora/lib/xsys.hh"

namespace ora {
namespace lib {
namespace fs {

using std::string;

//------------------------------------------------------------------------------

string
load_text(
  int fd)
{
  size_t const BLOCK = 256 * 1024;

  char buffer[BLOCK];
  string text;
  size_t num_read;
  while ((num_read = read(fd, buffer, BLOCK)) > 0)
    text += string(buffer, num_read);
  return text;
}


string
load_text(
  Filename const& filename)
{
  size_t const size = (size_t) stat(filename).st_size;
  char buffer[size];
  int const fd = xopen(filename, O_RDONLY);
  size_t const num_read = xread(fd, buffer, size);
  assert(num_read == size);
  xclose(fd);
  // FIXME: Do we need this copy?
  return string(buffer, size);
}


//------------------------------------------------------------------------------

}  // namespace fs
}  // namespace lib
}  // namespace ora

