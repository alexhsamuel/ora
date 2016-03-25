#include <cassert>
#include <cstring>
#include <string>

#include "file.hh"
#include "filename.hh"
#include "xsys.hh"

using std::string;

namespace alxs {
namespace fs {

//------------------------------------------------------------------------------

string
mode_as_str(
  mode_t mode)
{
  switch (mode) {
  case O_RDONLY                         : return "r";
  case O_WRONLY | O_CREAT               : return "w";
  case O_RDWR   | O_CREAT               : return "w+";
  case O_WRONLY | O_CREAT | O_APPEND    : return "a";
  case O_RDWR   | O_CREAT | O_APPEND    : return "a+";
  default:
    // FIXME: No good.  What do we do??
    assert(false);
  };
}


mode_t
mode_from_str(
  string const& str)
{
  if      (str == "r" ) return O_RDONLY;
  else if (str == "w" ) return O_WRONLY | O_CREAT;
  else if (str == "w+") return O_RDWR   | O_CREAT;
  else if (str == "a" ) return O_WRONLY | O_CREAT | O_APPEND;
  else if (str == "a+") return O_RDWR   | O_CREAT | O_APPEND;
  else
    throw ValueError(string("bad mode: ") + str);
}


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
  // FIXME: Do we need this copy?
  return string(buffer, size);
}


string
load_text_for_arg(
  string const& arg)
{
  if (arg == "-")
    return load_text(STDIN_FILENO);
  else
    return load_text(Filename(arg));
}


//------------------------------------------------------------------------------

}  // namespace fs
}  // namespace alxs

