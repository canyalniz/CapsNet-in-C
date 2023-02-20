project(CapsNet)
cmake_minimum_required(VERSION 3.12)
# Stop MSVC from complaining about insecure standard c functions
if (MSVC)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif ()

set(HEADER_FILES 
    include/capsule_inference.h
    include/h5_format.h
	include/image_input_format.h
)

set(SOURCE_FILES
    src/CapsNet_inference.c
    src/capsule_inference.c
	src/h5_format.c
	src/image_input_format.c
)

add_executable(CapsNet ${SOURCE_FILES} ${HEADER_FILES})
target_include_directories(CapsNet PRIVATE
    include
)

target_link_libraries(CapsNet m)

set_target_properties(CapsNet
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/.."
)
