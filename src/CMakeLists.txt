# Copyright (c) 2019 Antoine Tran Tan

cmake_minimum_required(VERSION 3.8)

file(GLOB_RECURSE sources
    "*.cpp"
)

add_executable(my_exe ${sources})

target_link_libraries(my_exe ${CONAN_LIBS})
