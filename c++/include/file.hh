#ifndef __FILE_HH__
#define __FILE_HH__

#include <string>

#include "filename.hh"

namespace alxs {
namespace fs {

//------------------------------------------------------------------------------

extern std::string mode_as_str(mode_t mode);
extern mode_t mode_from_str(std::string const& str);

extern std::string load_text(int fd);
extern std::string load_text(Filename const& filename);
extern std::string load_text_for_arg(std::string const& arg);

//------------------------------------------------------------------------------

}  // namespace fs
}  // namespace alxs

#endif  // #ifndef __FILE_HH__

