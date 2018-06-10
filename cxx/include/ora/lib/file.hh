#include <string>

#include "filename.hh"

namespace ora {
namespace lib {
namespace fs {

//------------------------------------------------------------------------------

extern std::string load_text(int fd);
extern std::string load_text(Filename const& filename);

/*
 * Line-by-line string input iterator adapter for istream.
 */
class LineIterator
: public std::iterator<std::input_iterator_tag, std::string>
{
public:

  /*
   * Constructs the end iterator.
   */
  LineIterator() : in_(nullptr), eol_('\n'), end_(true) {}

  LineIterator(
    std::istream* const in, 
    char const eol='\n') 
  : in_(in)
  , eol_(eol) 
  {
    advance();
  }

  LineIterator&
  operator++()
  {
    advance();
    return *this;
  }

  std::string operator*() const { return line_; }
  bool operator==(LineIterator const& i) const { return end_ && i.end_; }
  bool operator!=(LineIterator const& i) const { return !operator==(i); }

private:

  void
  advance()
  {
    line_.clear();
    if (end_)
      return;

    char buffer[1024];
    do {
      in_->getline(buffer, sizeof(buffer), eol_);
      line_ += buffer;
      if (in_->eof() || in_->bad()) {
        // End of file or failure, so mark the iterator as ended.
        end_ = true;
        break;
      }
      // Keep going to EOL.
    } while (in_->fail());
  }

  /* Input stream we're reading from.  */
  std::istream* const in_;

  /* The end of line character.  */
  char const eol_;

  /* True once the iterator has ended.  */
  bool end_ = false;

  /* The current line.  */
  std::string line_;

};


//------------------------------------------------------------------------------

}  // namespace fs
}  // namespace lib
}  // namespace ora

