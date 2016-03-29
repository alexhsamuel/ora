# Need 3.1 for target_compile_features
cmake_minimum_required(VERSION 3.1)
project(cron VERSION 1.0)

# TODO: Choose standard library from those available based on version
add_compile_options(-stdlib=libc++)

add_subdirectory(c++)
