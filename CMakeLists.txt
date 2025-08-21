cmake_minimum_required(VERSION 3.15)
project(cuMatch)

# C++
set(CMAKE_CXX_COMPILER /usr/bin/c++)
set(CMAKE_CXX_FLAGS "-Wall -O3 -lstdc++ -std=c++17 -fopenmp -D_FILE_OFFSET_BITS=64 -DNDEBUG") # release mode
# set(CMAKE_CXX_FLAGS "-Wall -O3 -lstdc++ -std=c++17 -fopenmp -D_FILE_OFFSET_BITS=64 -DDEBUG") # debugging mode

# CUDA
find_package(CUDA REQUIRED)

set(CUDA_HOST_COMPILER g++-7)
set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
list(APPEND CUDA_NVCC_FLAGS 
	#-gencode arch=compute_50,code=sm_50
	#-gencode arch=compute_52,code=sm_52
	#-gencode arch=compute_60,code=sm_60
	#-gencode arch=compute_61,code=sm_61
	#-gencode arch=compute_70,code=sm_70
	#-gencode arch=compute_75,code=sm_75
	-gencode arch=compute_80,code=sm_80
	-O3 -std=c++17 -Xcompiler -fopenmp)

# Boost
set(BOOST_ROOT "~/local/boost_1_82_0")
set(BOOST_INCLUDEDIR "~/local/include")
set(BOOST_LIBRARYDIR "~/local/lib")
find_package(Boost 1.82.0 REQUIRED)

# Nlohmann
include(FetchContent)
FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
FetchContent_MakeAvailable(json)
include_directories(${json_SOURCE_DIR}/include)

cmake_policy(SET CMP0015 NEW)
include_directories(${CMAKE_SOURCE_DIR}/include ${BOOST_INCLUDEDIR} ${CUDA_INCLUDE_DIRS})
link_directories(${CMAKE_SOURCE_DIR}/lib ${BOOST_LIBRARYDIR})

set(_ASH ash)
set(_GSTREAM gstream)
set(_CUMATCH cuMatch)
set(_LGF_GEN labeled_gridgen)

# cuMatch: LGF generator
file(GLOB_RECURSE _SOURCE ${CMAKE_SOURCE_DIR}/source/${_LGF_GEN}/*)
add_executable(${_LGF_GEN} ${_SOURCE})
target_link_libraries(${_LGF_GEN} ${_GSTREAM} ${_ASH} pthread boost_fiber boost_context boost_filesystem boost_system boost_exception)

# cuMatch
file(GLOB_RECURSE _SOURCE ${CMAKE_SOURCE_DIR}/source/${_CUMATCH}/*)
cuda_add_executable(${_CUMATCH} ${_SOURCE})
target_link_libraries(${_CUMATCH} ${_GSTREAM} ${_ASH} pthread boost_fiber boost_context boost_filesystem boost_system boost_exception)

# tools
foreach(_ITEM csv_relabeler)
	file(GLOB_RECURSE _SOURCE ${CMAKE_SOURCE_DIR}/source/tools/${_ITEM}/*)
	add_executable(${_ITEM} ${_SOURCE})
	target_link_libraries(${_ITEM} ${_ASH} ${_GSTREAM} pthread boost_fiber boost_context boost_filesystem boost_system boost_exception)
endforeach()
