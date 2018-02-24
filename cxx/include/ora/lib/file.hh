#include <string>

#include "filename.hh"

namespace ora {
namespace lib {
namespace fs {

//------------------------------------------------------------------------------

extern std::string load_text(int fd);
extern std::string load_text(Filename const& filename);

//------------------------------------------------------------------------------

}  // namespace fs
}  // namespace lib
}  // namespace ora

