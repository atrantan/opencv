# Copyright (c) 2019 Antoine Tran Tan

cmake_minimum_required(VERSION 3.7.2)

file(GLOB_RECURSE sources
    "*.cpp"
)

add_executable(my_exe ${sources})

target_link_libraries(my_exe ${CONAN_LIBS})

add_custom_command(TARGET my_exe PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/docs ${CMAKE_CURRENT_BINARY_DIR})
