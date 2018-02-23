cmake_minimum_required(VERSION 3.5.1)
project(underactuated)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE STRING
    "Prefix for installation of sub-packages (note: required during build!)"
    FORCE)
endif()

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/modules")

set(PYBIND11_INSTALL True)
set(PYTHON_EXECUTABLE "python2.7")
set(PYBIND11_INCLUDE_DIR "${CMAKE_INSTALL_PREFIX}/include")
add_subdirectory(pybind11)

find_package(drake CONFIG REQUIRED)

add_library(battleship_utils battleship_utils.cc)
set_target_properties(battleship_utils PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
target_link_libraries(battleship_utils
    drake::drake)

pybind11_add_module(battleship_utils_py MODULE battleship_utils_py.cc)
target_link_libraries(battleship_utils_py PRIVATE battleship_utils drake::drake)
target_include_directories(battleship_utils PRIVATE ${PYBIND11_INCLUDE_DIR})
set_target_properties(battleship_utils_py PROPERTIES PREFIX "" SUFFIX ".so")
INSTALL(TARGETS battleship_utils_py DESTINATION lib/python2.7/site-packages/)