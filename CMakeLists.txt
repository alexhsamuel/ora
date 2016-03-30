# Need 3.1 for target_compile_features
cmake_minimum_required(VERSION 3.1)
project(cron VERSION 1.0)
enable_testing()

add_custom_target(check
  ${CMAKE_COMMAND} -E env CTEST_OUTPUT_ON_FAILURE=1 GTEST_COLOR=1
  ${CMAKE_CTEST_COMMAND} -C $<CONFIG> --verbose
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

# TODO: Choose standard library from those available based on version
add_compile_options(-stdlib=libc++)

add_subdirectory(c++)
