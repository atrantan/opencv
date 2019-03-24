# Copyright (c) 2019 Antoine Tran Tan

cmake_minimum_required (VERSION 3.8)
project ("TP_OpenCV")

# Utilisation au minimum de C++14
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Setup de Conan
include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

# Ajout de nos dossiers
include_directories(include)
add_subdirectory(src)
