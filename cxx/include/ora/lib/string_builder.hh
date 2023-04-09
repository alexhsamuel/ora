#pragma once

#include <cassert>
#include <cstring>
#include <string>

namespace ora {
namespace lib {

//------------------------------------------------------------------------------

class StringBuilder
{
public:

  StringBuilder(
    size_t hint=32)
    : length_(0),
      size_(0),
      buffer_(nullptr)
  {
    assert(hint > 0);
    resize(hint);
  }

  size_t length() const { return length_; }
  operator char const*() const { return buffer_; }
  std::string str() const { return std::string(buffer_, length_); }

  ~StringBuilder()
  {
    if (buffer_ != nullptr)
      free(buffer_);
  }

  StringBuilder&
  operator<<(
    char c)
  {
    maybe_resize(1);
    buffer_[length_++] = c;
    return *this;
  }

  StringBuilder&
  operator<<(
    std::string const& str)
  {
    size_t const length = str.length();
    maybe_resize(length);
    memcpy(buffer_ + length_, str.c_str(), str.length());
    length_ += length;
    return *this;
  }

  StringBuilder&
  pad(
    size_t length,
    char pad=' ')
  {
    maybe_resize(length);
    memset(buffer_ + length_, pad, length);
    length_ += length;
    return *this;
  }

  StringBuilder&
  format(
    uint64_t value,
    size_t width=0,
    char fill=' ')
  {
    maybe_resize(width);

    // Count the number of digits needed to represent the value.  At least one
    // digit is required.
    size_t digits = 1;
    for (uint64_t rem = value / 10; rem > 0; digits++, rem /= 10)
      ;

    if (digits > width)
      // Expand the width to accommodate the entire value.
      width = digits;
    else if (digits < width)
      // Pad.
      for (size_t i = 0; i < width - digits; ++i)
        *this << fill;

    // Generate the digits, from least to most significant.
    // Buffer size: log_10(1 << 64) ~ 19.3
    char buf[20];
    assert(digits <= sizeof(buf));
    for (ssize_t i = digits - 1; i >= 0; --i) {
      buf[i] = '0' + value % 10;
      value /= 10;
    }
    assert(value == 0);
    // Now produce the digits in order.
    for (size_t i = 0; i < digits; ++i)
      *this << buf[i];

    return *this;
  }

  StringBuilder&
  rstrip(
    char const c)
  {
    while (length_ > 0 && buffer_[length_ - 1] == c)
      --length_;
    return *this;
  }

private:

  void
  maybe_resize(
    size_t increment)
  {
    size_t const new_size = length_ + increment + 1;
    if (new_size > size_)
      resize(std::max(new_size, length_ * 2));
  }

  void
  resize(
    size_t new_size)
  {
    assert(new_size > length_);
    buffer_ = (char*) realloc(buffer_, new_size);
    assert(buffer_ != nullptr);
    size_ = new_size;
  }
  
  size_t length_;
  size_t size_;
  char* buffer_;

};


//------------------------------------------------------------------------------

}  // namespace lib
}  // namespace ora

