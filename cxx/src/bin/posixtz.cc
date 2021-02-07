#include <iostream>

#include "ora.hh"

using namespace ora;

//------------------------------------------------------------------------------

int
main(
  int const argc,
  char const* const* argv)
{
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " TZSPEC\n";
    return EXIT_FAILURE;
  }

  PosixTz tz;
  try {
    tz = parse_posix_time_zone(argv[1]);
  }
  catch (ora::lib::FormatError& err) {
    std::cerr << "Error: " << err.what() << "\n";
    return EXIT_FAILURE;
  }
  std::cout << tz << "\n";
  return EXIT_SUCCESS;
}


  
