file(GLOB SOURCES "src/*.cc")
add_library(cron ${SOURCES})
set_property(TARGET cron PROPERTY CXX_STANDARD 11)
target_compile_features(cron
  PUBLIC cxx_relaxed_constexpr
  )
target_include_directories(cron
  PUBLIC include
  )
