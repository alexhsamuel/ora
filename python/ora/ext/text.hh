#pragma once

#include <cassert>
#include <iostream>
#include <string>

//------------------------------------------------------------------------------

namespace {

inline bool
within(
  unsigned char min, 
  unsigned char val, 
  unsigned char max)
  noexcept
{
  return min <= val && val <= max;
}


/*
 * Rounds 'val' to the nearest integer using banker's rounding.  
 */
inline long 
round(
  double const val)
  noexcept
{
  // FIXME: Broken for val > MAX_LONG.
  long const i = (long) val;
  double const r = val - i;
  return 
    i + (val > 0
    ? (r <  0.5 || (i % 2 == 0 && r ==  0.5) ? 0 :  1)
    : (r > -0.5 || (i % 2 == 0 && r == -0.5) ? 0 : -1));
}


}  // anonymous namespace

//------------------------------------------------------------------------------

namespace ora {
namespace py {

using std::string;

constexpr char const* ELLIPSIS = "\u2026";
constexpr char ANSI_ESCAPE = '\x1b';

/*
 * Advances an iterator on a UTF-8 string by one code point.
 *
 * FIXME: Take an end parameter.
 */
inline bool
next_utf8(
  string::const_iterator& i)
  noexcept
{
  unsigned char c = *i++;
  if ((c & 0xc0) == 0xc0) {
    // It's multibyte.  The number of bytes is the number of MSB's before 
    // the first zero.
    // FIXME: Improve this.
    ++i;
    if ((c & 0xe0) == 0xe0) {
      ++i;
      if ((c & 0xf0) == 0xf0) {
        ++i;
        if ((c & 0xf8) == 0xf8) {
          ++i;
          if ((c & 0xfc) == 0xfc)
            ++i;
        }
      }
    }
  }
  return true;
}


/*
 * Advances an iterator past an ANSI escape sequence, if at one.
 */
inline bool
skip_ansi_escape(
  string::const_iterator& i, 
  string::const_iterator const& end)
  noexcept
{
  assert(i != end);
  if (*i == ANSI_ESCAPE) {
    ++i;
    if (i != end && *i++ == '[') 
      // Got CSI.  Read until we pass a final byte.
      while (i != end && !within(64, *i++, 126))
        ;
    else
      // Assume single-character escape.
      ;
    return true;
  }
  else
    return false;
}


/*
 * Returns the number of code points in a UTF-8-encoded string, skipping
 * escape sequences.
 */
inline size_t
string_length(
  string const& str)
  noexcept
{
  size_t length = 0;
  auto const& end = str.end();
  // FIXME: Problem if the last code point is malformed.
  // Count characters.
  for (auto i = str.begin(); i != end; ) 
    if (skip_ansi_escape(i, end))
      ;
    else {
      ++length;
      next_utf8(i);
    }
  return length;
}


/*
 * Concatenates copies of `str` up to `length`.  If `length` is not divisible
 * by the length of `str`, the last copy is partial.
 */
inline string
fill(
  string const& str,
  size_t const length)
{
  size_t const str_len = string_length(str);
  assert(str_len > 0);
  if (str.length() == 1)
    return string(length, str[0]);
  else {
    string result;
    result.reserve(length);
    // Concatenate whole copies.
    size_t l = length;
    while (l >= str_len) {
      result += str;
      l -= str_len;
    }
    if (l > 0) {
      // Concatenate a partial copy.
      auto i = str.begin();
      for (; l > 0; --l)
        next_utf8(i);
      result.append(str.begin(), i);
    }
    assert(string_length(result) == length);
    return result;
  }
}


float constexpr PAD_POS_LEFT_JUSTIFY   = 1.0;
float constexpr PAD_POS_CENTER         = 0.5;
float constexpr PAD_POS_RIGHT_JUSTIFY  = 0.0;


/**
 * Pads a string in to fixed length on one or both sides.
 *
 * If position=1, the string is padded at the right, thus left-justified.  If
 * position=0, the string is padded at the left, thus right-justified.  If
 * position=0.5, the string is padded equally on both sides, thus centered.
 */
inline string
pad(
  string const& str,
  size_t const length,
  string const& pad=" ",
  float const pos=PAD_POS_LEFT_JUSTIFY)
{
  assert(string_length(pad) > 0);
  assert(0 <= pos);
  assert(pos <= 1);
  size_t const str_len = string_length(str);
  if (str_len < length) {
    size_t const pad_len = length - str_len;
    size_t const left_len = (size_t) round((1 - pos) * pad_len);
    string const result = 
      fill(pad, left_len) + str + fill(pad, pad_len - left_len);
    assert(string_length(result) == length);
    return result;
  }
  else
    return str;
}


/**
 * Trims a string to a fixed length by eliding characters and replacing them
 * with an ellipsis.
 */
inline string
elide(
  string const& str,
  size_t const max_length,
  string const& ellipsis=ELLIPSIS,
  float const pos=1)
{
  size_t const ellipsis_len = string_length(ellipsis);
  assert(max_length >= ellipsis_len);
  assert(0 <= pos);
  assert(pos <= 1);

  size_t const length = string_length(str);
  if (length <= max_length)
    return str;
  else {
    size_t const keep   = max_length - ellipsis_len;
    size_t const nleft  = (size_t) round(pos * keep);
    size_t const nright = keep - nleft;
    string elided;
    if (nleft > 0)
      elided += str.substr(0, nleft);
    elided += ellipsis;
    if (nright > 0)
      elided += str.substr(length - nright);
    assert(string_length(elided) == max_length);
    return elided;
  }
}


/**
 * Either pads or elides a string to achieve a fixed length.
 */
inline string
palide(
  string const& str,
  size_t const length,
  string const& ellipsis=ELLIPSIS,
  string const& pad=" ",
  float const elide_pos=1,
  float pad_pos=1)
{
  return ora::py::pad(
    elide(str, length, ellipsis, elide_pos), length, pad, pad_pos);
}


//------------------------------------------------------------------------------

}  // namespace py
}  // namespace ora

