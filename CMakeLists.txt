cmake_minimum_required(VERSION 3.5.1)

option(LOG_ALL "Print all logging levels" ON)

# Notice: use cmake with `-DCMAKE_GENERATOR_PLATFORM=x64`.

# ------------------------------------------------------------------------------
# ----------------------------------- Common -----------------------------------
# ------------------------------------------------------------------------------
project(FaceCap CXX)
set(CMAKE_CXX_STANDARD 11)  # C++ 11

# Print make commands for debug
# set(CMAKE_VERBOSE_MAKEFILE 1)

# Set default build type
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# cmake modules
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/sanitizers)
#find_package(Sanitizers) # Address sanitizer (-DSANITIZE_ADDRESS=ON)

# Set output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Warning options
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wconversion")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} --pedantic -Wall -Wextra -Wcast-align -Wcast-qual -Wctor-dtor-privacy -Wdisabled-optimization -Wformat=2 -Winit-self -Wlogical-op -Wmissing-declarations -Wmissing-include-dirs -Wnoexcept -Wold-style-cast -Woverloaded-virtual -Wredundant-decls -Wshadow -Wsign-conversion -Wsign-promo -Wstrict-null-sentinel -Wstrict-overflow=5 -Wundef -Wno-unknown-pragmas")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} /W4")
endif()


# ------------------------------------------------------------------------------
# ---------------------------- Project Specificated ----------------------------
# ------------------------------------------------------------------------------

# ----------------------------- Internal libraries -----------------------------
# [OpenCV]
# Use local copy of prebuilt OpenCV library for Visual Studio 2017
set(OpenCV_DIR "${CMAKE_SOURCE_DIR}/third_party/opencv/win64_vc15/opencv-3.4.0/build/")
include_directories(${CMAKE_SOURCE_DIR}/third_party/opencv/win64_vc15/opencv-3.4.0/include) # 上記ディレクトリの下のincludeディレクトリ

find_package(OpenCV REQUIRED)

# Internal includes
include_directories(
    ${CMAKE_SOURCE_DIR}/third_party/
    ${CMAKE_SOURCE_DIR}/third_party/glfw/include/
    ${CMAKE_SOURCE_DIR}/third_party/
    ${CMAKE_SOURCE_DIR}/third_party/AMDSDK/3.0/include/
    ${CMAKE_SOURCE_DIR}/third_party/KinectSDK/v1.8/inc/
    ${CMAKE_SOURCE_DIR}/third_party/HuaweiLibrary/
    ${OpenCV_INCLUDE_DIRS}
)

# [HuaweiLibrary]
add_library (HuaweiLibrary STATIC
    ${CMAKE_SOURCE_DIR}/third_party/HuaweiLibrary/hwSensorManager.cpp
    ${CMAKE_SOURCE_DIR}/third_party/HuaweiLibrary/hwKinectManager.cpp
    ${CMAKE_SOURCE_DIR}/third_party/HuaweiLibrary/hwOrbecManager.cpp
)
set(EXTRA_LIBS ${EXTRA_LIBS} HuaweiLibrary)

# [glfw]
set(GLFW_BUILD_EXAMPLES OFF)
set(GLFW_BUILD_TESTS OFF)
add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/glfw)
set(EXTRA_LIBS ${EXTRA_LIBS} glfw)

# [OpenGL]
FIND_PATH(OPENGL_INCLUDE_DIR GL/glew.h /third_party/AMDSDK/3.0/include)
link_directories(${CMAKE_SOURCE_DIR}/third_party/AMDSDK/3.0/lib/x86_64/)
set(EXTRA_LIBS ${EXTRA_LIBS} glut64)
set(EXTRA_LIBS ${EXTRA_LIBS} glew64)
#find_package(OpenGL REQUIRED)
#set(EXTRA_LIBS ${EXTRA_LIBS} ${OPENGL_LIBRARY} glew)

# [Kinect]
link_directories(${CMAKE_SOURCE_DIR}/third_party/KinectSDK/v1.8/lib/amd64/)
set(EXTRA_LIBS ${EXTRA_LIBS} Kinect10)

# [Intraface]
link_directories(${CMAKE_SOURCE_DIR}/third_party/intraface/lib/)
set(EXTRA_LIBS ${EXTRA_LIBS} intraface_core131)

# ----------------------------------- Sources ----------------------------------
add_library(diego_lib STATIC
    ${CMAKE_SOURCE_DIR}/src/App.cpp
    ${CMAKE_SOURCE_DIR}/src/FaceCap.cpp
    ${CMAKE_SOURCE_DIR}/src/Mesh.cpp
    ${CMAKE_SOURCE_DIR}/src/SLIC.cpp
    ${CMAKE_SOURCE_DIR}/src/stdafx.cpp
    ${CMAKE_SOURCE_DIR}/src/Utility.cpp
)
set(EXTRA_LIBS ${EXTRA_LIBS} diego_lib)

#target_link_libraries(facecap ${EXTRA_LIBS})

#add_sanitizers(facecap)

# --------------------------------- Executables --------------------------------
add_executable(facecap_pc
    ${CMAKE_SOURCE_DIR}/pc/main.cpp
)
target_link_libraries(facecap_pc ${EXTRA_LIBS})
target_link_libraries(facecap_pc ${OpenCV_LIBS})

#add_sanitizers(facecap_pc)

# --------------------------------- Post Build ---------------------------------
if (WIN32)
    # Copy OpenCV dlls
    file(GLOB OPENCV_DLLS
        "${CMAKE_SOURCE_DIR}/third_party/opencv/win64_vc15/opencv-3.4.0/build/bin/Release/opencv_*.dll")
    add_custom_command(TARGET facecap_pc
                       POST_BUILD
                       COMMAND ${CMAKE_COMMAND}
                       ARGS -E copy_if_different ${OPENCV_DLLS} ${CMAKE_CURRENT_BINARY_DIR}
                      )

    # Copy OpenCV dlls
    file(GLOB OPENCV_DB_DLLS
        "${CMAKE_SOURCE_DIR}/third_party/opencv/win64_vc15/opencv-3.4.0/build/bin/Debug/opencv_*.dll")
    add_custom_command(TARGET facecap_pc
                       POST_BUILD
                       COMMAND ${CMAKE_COMMAND}
                       ARGS -E copy_if_different ${OPENCV_DB_DLLS} ${CMAKE_CURRENT_BINARY_DIR}
                      )

    # Copy Intraface dlls
    file(GLOB INTRAFACE_DLL "${CMAKE_SOURCE_DIR}/third_party/intraface/bin/*.dll")
    add_custom_command(TARGET facecap_pc
                       POST_BUILD
                       COMMAND ${CMAKE_COMMAND}
                       ARGS -E copy_if_different ${INTRAFACE_DLL} ${CMAKE_CURRENT_BINARY_DIR}
                      )

    # Copy OPENGL dlls
    file(GLOB OPENGL_DLL "${CMAKE_SOURCE_DIR}/third_party/AMDSDK/3.0/bin/x86_64/gl*.dll")
    add_custom_command(TARGET facecap_pc
                       POST_BUILD
                       COMMAND ${CMAKE_COMMAND}
                       ARGS -E copy_if_different ${OPENGL_DLL} ${CMAKE_CURRENT_BINARY_DIR}
                      )
endif()