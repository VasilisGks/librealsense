#  minimum required cmake version: 3.1.0
cmake_minimum_required(VERSION 3.1.0)
 
project(RealsenseExamplesCapture)

# Save the command line compile commands in the build output
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

#NEW


include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
endif()


find_package(PCL 1.9.0 REQUIRED)

if (PCL_FOUND)
  include_directories(${PCL_INCLUDE_DIRS})
  link_directories(${PCL_LIBRARY_DIRS})
  add_definitions(${PCL_DEFINITIONS})
endif()
#END NEW

if(BUILD_GRAPHICAL_EXAMPLES)
    add_executable(rs-custom  rs-custom.cpp ../example.hpp)
    target_link_libraries(rs-custom ${DEPENDENCIES})
    include_directories(rs-custom ../ ../../third-party/tclap/include)
    set_target_properties (rs-custom PROPERTIES FOLDER "Examples")

    install(TARGETS rs-custom RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

if(BUILD_GRAPHICAL_EXAMPLES)
    add_executable(without_filters  without_filters.cpp ../example.hpp)
    target_link_libraries(without_filters ${DEPENDENCIES})
    include_directories(without_filters ../ ../../third-party/tclap/include)
    set_target_properties (without_filters PROPERTIES FOLDER "Examples")

    install(TARGETS without_filters RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

target_link_libraries(rs-custom ${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES} ${PCL_FILTERS_LIBRARIES})
target_link_libraries(rs-custom ${PCL_IO_LIBRARIES})
