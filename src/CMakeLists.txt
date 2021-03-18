file(GLOB_RECURSE sources
    "*.c"
    "*.cpp"
)

add_executable(my_exe ${sources})

# Link to math library if compiler is gnu
if ("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
    list(APPEND libraries m)
endif()

# Link to user libraries (if there are any)
if(libraries)
    target_link_libraries(my_exe ${libraries})
endif()

target_link_libraries(my_exe ${OpenCV_LIBS})

install( TARGETS my_exe
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
