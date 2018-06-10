#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "ora/lib/string.hh"
#include "ora/calendar.hh"

namespace ora {

using namespace ora::lib;
using date::Date;

//------------------------------------------------------------------------------
// Functions
//------------------------------------------------------------------------------

namespace {

// FIXME: Elsewhere.
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
    if (end_)
      return;

    line_.clear();
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

  std::istream* const in_;
  char const eol_;
  bool end_ = false;
  std::string line_;

};


}  // anonymous namespace


Calendar
load_calendar(
  fs::Filename const& filename)
{
  std::ifstream in((char const*) filename);
  return parse_calendar(LineIterator(&in), LineIterator());
}


Calendar
make_const_calendar(
  Range<Date> const range,
  bool const contains)
{
  auto dates = std::vector<bool>(range.max - range.min + 1, contains);
  return {range.min, std::move(dates)};
}


Calendar
make_weekday_calendar(
  Range<Date> const range,
  bool const mask[7])
{
  auto dates = std::vector<bool>();
  auto const length = range.max - range.min + 1;
  dates.reserve(length);
  for (auto i = 0; i < length; ++i)
    dates.push_back(mask[get_weekday(range.min + i)]);
  return {range.min, std::move(dates)};
}


//------------------------------------------------------------------------------

}  // namespace ora


