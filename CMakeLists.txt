cmake_minimum_required(VERSION 3.10)
project(FeatureMatcher CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenCV >= 4.4 (SIFT is in the main features2d module since 4.4). OpenCV 5 is also
# accepted; its config rejects a "4.x" version request and renames modules
# (features2d -> features, calib3d -> calib/geometry), so the check is done by hand.
# On Windows set OpenCV_DIR to e.g. C:/opencv/build
find_package(OpenCV REQUIRED)
if(OpenCV_VERSION VERSION_LESS 4.4)
    message(FATAL_ERROR "OpenCV >= 4.4 is required, found ${OpenCV_VERSION}")
endif()
message(STATUS "Using OpenCV ${OpenCV_VERSION}")

add_executable(FeatureMatcher
    src/main.cpp
    src/features.cpp
    src/evaluation.cpp)

target_include_directories(FeatureMatcher PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(FeatureMatcher PRIVATE ${OpenCV_LIBS})

if(MSVC)
    target_compile_options(FeatureMatcher PRIVATE /W3 /utf-8)
else()
    target_compile_options(FeatureMatcher PRIVATE -Wall -Wextra)
endif()
