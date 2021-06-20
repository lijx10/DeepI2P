cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_STANDARD 11)

project(FrustumRegistration)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall")

add_subdirectory(pybind11)
find_package(Eigen3)
find_package(Ceres REQUIRED)

set(MY_HEADERS
    ${EIGEN3_INCLUDE_DIR}
    src/utils.hpp
    src/registration_2d.hpp
    src/registration_3d.hpp
)

INCLUDE_DIRECTORIES(
    ${MY_HEADERS}
    ${CERES_INCLUDE_DIRS}
)

pybind11_add_module(${PROJECT_NAME}
    src/registration.cpp
    ${MY_HEADERS}
)
target_link_libraries(${PROJECT_NAME}
    ${CERES_LIBRARIES}
)
