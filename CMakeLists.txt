cmake_minimum_required(VERSION 3.10)
project(obscure)

set(PLUGIN_NAME "obscure")
set(OBS_PATH "libs/obs-studio")

# set(CMAKE_C_COMPILER "/usr/bin/gcc-12")
# set(CMAKE_CXX_COMPILER "/usr/bin/g++-12")
set(CMAKE_C_COMPILER "/usr/bin/gcc-13")
set(CMAKE_CXX_COMPILER "/usr/bin/g++-13")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_COMPILER "/opt/cuda/bin/nvcc")
set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)

include_directories(${OBS_PATH}/libobs)
link_directories(${OBS_PATH}/build/libobs)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(CAFFE2_USE_CUDNN ON)
set(Torch_DIR "${CMAKE_SOURCE_DIR}/libs/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_library(${PLUGIN_NAME} MODULE 
    src/obscure.cpp 
)

configure_file(${CMAKE_SOURCE_DIR}/models/yolov8l-seg_288-480.torchscript
              ${CMAKE_BINARY_DIR}/obscure/bin/64bit/yolov8l-seg_288-480.torchscript COPYONLY)

target_include_directories(${PLUGIN_NAME} PRIVATE
    ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(${PLUGIN_NAME} 
    obs 
    ${OpenCV_LIBS}
    ${TORCH_LIBRARIES}
)

if(UNIX)
    set_target_properties(${PLUGIN_NAME} PROPERTIES
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH "$ORIGIN/lib"
        INSTALL_RPATH_USE_LINK_PATH TRUE
    )
endif()

set_target_properties(${PLUGIN_NAME} PROPERTIES
    PREFIX ""
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/obscure/bin/64bit"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/obscure/bin/64bit"
)

add_custom_command(TARGET ${PLUGIN_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/obscure/bin/64bit/lib"
    COMMAND ${CMAKE_COMMAND} -E copy
        "${CMAKE_SOURCE_DIR}/libs/libtorch/lib/libc10.so"
        "${CMAKE_BINARY_DIR}/obscure/bin/64bit/lib/libc10.so"
    COMMAND ${CMAKE_COMMAND} -E copy
        "${CMAKE_SOURCE_DIR}/libs/libtorch/lib/libcublasLt-44a813aa.so.12"
        "${CMAKE_BINARY_DIR}/obscure/bin/64bit/lib/libcublasLt-44a813aa.so.12"
    COMMAND ${CMAKE_COMMAND} -E copy
        "${CMAKE_SOURCE_DIR}/libs/libtorch/lib/libgomp-98b21ff3.so.1"
        "${CMAKE_BINARY_DIR}/obscure/bin/64bit/lib/libgomp-98b21ff3.so.1"
    COMMAND ${CMAKE_COMMAND} -E copy
        "${CMAKE_SOURCE_DIR}/libs/libtorch/lib/libtorch_cuda.so"
        "${CMAKE_BINARY_DIR}/obscure/bin/64bit/lib/libtorch_cuda.so"
    COMMAND ${CMAKE_COMMAND} -E copy
        "${CMAKE_SOURCE_DIR}/libs/libtorch/lib/libc10_cuda.so"
        "${CMAKE_BINARY_DIR}/obscure/bin/64bit/lib/libc10_cuda.so"
    COMMAND ${CMAKE_COMMAND} -E copy
        "${CMAKE_SOURCE_DIR}/libs/libtorch/lib/libcudart-8774224f.so.12"
        "${CMAKE_BINARY_DIR}/obscure/bin/64bit/lib/libcudart-8774224f.so.12"
    COMMAND ${CMAKE_COMMAND} -E copy
        "${CMAKE_SOURCE_DIR}/libs/libtorch/lib/libtorch.so"
        "${CMAKE_BINARY_DIR}/obscure/bin/64bit/lib/libtorch.so"
    COMMAND ${CMAKE_COMMAND} -E copy
        "${CMAKE_SOURCE_DIR}/libs/libtorch/lib/libcublas-4cc45526.so.12"
        "${CMAKE_BINARY_DIR}/obscure/bin/64bit/lib/libcublas-4cc45526.so.12"
    COMMAND ${CMAKE_COMMAND} -E copy
        "${CMAKE_SOURCE_DIR}/libs/libtorch/lib/libcudnn.so.9"
        "${CMAKE_BINARY_DIR}/obscure/bin/64bit/lib/libcudnn.so.9"
    COMMAND ${CMAKE_COMMAND} -E copy
        "${CMAKE_SOURCE_DIR}/libs/libtorch/lib/libtorch_cpu.so"
        "${CMAKE_BINARY_DIR}/obscure/bin/64bit/lib/libtorch_cpu.so"
)

message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "CUDA Host Compiler: ${CMAKE_CUDA_HOST_COMPILER}")
message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER}")

set(TARGET_DIR "$ENV{HOME}/.config/obs-studio/plugins")
set(EXPORTED_PROJECT_DIR "${CMAKE_BINARY_DIR}/obscure")

add_custom_command(
    TARGET obscure POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different "${EXPORTED_PROJECT_DIR}" "${TARGET_DIR}/obscure"
    COMMENT "Installing 'obscure' to ${TARGET_DIR}"
)