#pragma once

#include <string>

#include "filename.hh"
#include "iter.hh"

namespace ora {
namespace lib {
namespace fs {

//------------------------------------------------------------------------------

extern std::string load_text(int fd);
extern std::string load_text(Filename const& filename);

/*
 * Line-by-line string input iterator adapter for istream.
 */
class LineIter
: public Iter<std::string>
{
public:

  LineIter(
    std::istream& in, 
    char const delim='\n')
  : in_(in)
  , delim_(delim)
  {
  }

  virtual ~LineIter() override = default;

  virtual optional<std::string> 
  next()
    override
  {
    if (end_)
      return {};

    auto line = std::string();
    std::getline(in_, line, delim_);

    if (in_.eof() || in_.bad()) {
      end_ = true;
      // We don't consider there to be an empty final line if the file ends with
      // a delimiter character.
      if (line.empty())
        return {};
    }

    return line;
  }

private:

  /* Input stream we're reading from.  */
  std::istream& in_;

  /* The end of line delimiter character.  */
  char const delim_;

  /* True once the iterator has ended.  */
  bool end_ = false;

};


//------------------------------------------------------------------------------

}  // namespace fs
}  // namespace lib
}  // namespace ora

