#include <cstdlib>
#include <iostream>

#include "ora/tzfile.hh"

using namespace ora;

using aslib::fs::Filename;

//------------------------------------------------------------------------------

int
main(
  int const argc,
  char const* const* const argv)
{
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " FILE\n";
    return EXIT_FAILURE;
  }

  auto const tz_file = TzFile::load(Filename{argv[1]});
  std::cout << tz_file << std::flush;
  return EXIT_SUCCESS;
}
