file(GLOB SOURCES "src/*.cc")
add_library(cron ${SOURCES})
set_property(TARGET cron PROPERTY CXX_STANDARD 11)
target_compile_features(cron PUBLIC cxx_relaxed_constexpr)
target_include_directories(cron PUBLIC include)

install(TARGETS cron DESTINATION lib)
install(DIRECTORY include/cron DESTINATION include)

# Download and unpack googletest at configure time
set(GTEST_DOWNLOAD_DIR ${CMAKE_BINARY_DIR}/googletest-download)
configure_file(CMakeLists.txt.googletest
  ${GTEST_DOWNLOAD_DIR}/CMakeLists.txt)
execute_process(
  COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
  WORKING_DIRECTORY ${GTEST_DOWNLOAD_DIR})
execute_process(
  COMMAND ${CMAKE_COMMAND} --build .
  WORKING_DIRECTORY ${GTEST_DOWNLOAD_DIR})

# Prevent GoogleTest from overriding our compiler/linker options
# when building with Visual Studio
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Add googletest directly to our build. This adds
# the following targets: gtest, gtest_main, gmock
# and gmock_main
add_subdirectory(${CMAKE_BINARY_DIR}/googletest-src
                 ${CMAKE_BINARY_DIR}/googletest-build
                 EXCLUDE_FROM_ALL)

file(GLOB TESTS "test/*.cc")
add_executable(cron-gtest ${TESTS})
target_link_libraries(cron-gtest cron gtest_main)
add_test(all-tests cron-gtest)
