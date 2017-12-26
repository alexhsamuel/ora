#include <string>

#include "aslib/exc.hh"
#include "ora.hh"

namespace ora {

//------------------------------------------------------------------------------

HmsDaytime
parse_iso_daytime(
  std::string const& text)
  noexcept
{
  char* end;
  auto colon = text.find(':');
  if (!(colon == 1 || colon == 2))
    return {};  // invalid
  Hour const hour = strtoul(&text[0], &end, 10);
  if (end != &text[colon])
    return {};  // invalid
  auto i = colon + 1;

  colon = text.find(':', i);
  if (colon != i + 2)
    return {};  // invalid
  Minute const minute = strtoul(&text[i], &end, 10);
  if (end != &text[colon])
    return {};  // invalid
  i = colon + 1;

  Second const second = strtod(&text[i], &end);
  if (end != &text[text.length()])
    return {};  // invalid

  return {hour, minute, second};
}


//------------------------------------------------------------------------------

}  // namespace ora

