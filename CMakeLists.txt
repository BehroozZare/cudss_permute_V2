#CMake minimum requirement 
#To run with cuDSS:
#>> cmake -DCMAKE_PREFIX_PATH="C:\Program Files\NVIDIA cuDSS\v0.6\lib\12\cmake" ..

cmake_minimum_required(VERSION 3.23 FATAL_ERROR)

#Project name 
set(project_name "TestcuDSS")#<TODO> change this to the name of your project 
project(${project_name} LANGUAGES CXX C CUDA)  

#toggle between building a shared or static library
option(MY_BUILD_SHARED_LIBS "Build as shared library" OFF)

#default build type is Release
if (CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE Release)
endif ()

# Direct all output to /bin directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)

# Auto-detect GPU architecture
include("cmake/AutoDetectCudaArch.cmake")
include("cmake/metis.cmake")
include("cmake/eigen.cmake")
include("cmake/cli11.cmake")
include("cmake/parth.cmake")

# CUDA and C++ compiler flags
# C++ compiler flags (config-aware)
set(cxx_flags
		# MSVC
		$<$<AND:$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Debug>>:
		/openmp /std:c++17 /Zi /Od>
		$<$<AND:$<CXX_COMPILER_ID:MSVC>,$<NOT:$<CONFIG:Debug>>>:
		/openmp /std:c++17>  # add /O2 here if you want

		# GCC
		$<$<AND:$<CXX_COMPILER_ID:GNU>,$<CONFIG:Debug>>:
		-Wall -m64 -fopenmp -std=c++17 -O0 -g3>
		$<$<AND:$<CXX_COMPILER_ID:GNU>,$<NOT:$<CONFIG:Debug>>>:
		-Wall -m64 -fopenmp -std=c++17 -O3>

		# Clang
		$<$<AND:$<CXX_COMPILER_ID:Clang>,$<CONFIG:Debug>>:
		-Wall -m64 -fopenmp -std=c++17 -O0 -g3>
		$<$<AND:$<CXX_COMPILER_ID:Clang>,$<NOT:$<CONFIG:Debug>>>:
		-Wall -m64 -fopenmp -std=c++17 -O3>
)

set(MSVC_XCOMPILER_FLAGS "/openmp /std:c++17")

# CUDA compiler flags (config-aware)
set(cuda_flags
		# Host compiler side for CUDA
		$<$<AND:$<CXX_COMPILER_ID:GNU>,$<CONFIG:Debug>>:
		-Xcompiler -Wall -fopenmp -O0 -g>
		$<$<AND:$<CXX_COMPILER_ID:GNU>,$<NOT:$<CONFIG:Debug>>>:
		-Xcompiler -Wall -fopenmp -O3>

		$<$<AND:$<CXX_COMPILER_ID:Clang>,$<CONFIG:Debug>>:
		-Xcompiler -Wall -fopenmp -O0 -g>
		$<$<AND:$<CXX_COMPILER_ID:Clang>,$<NOT:$<CONFIG:Debug>>>:
		-Xcompiler -Wall -fopenmp -O3>

		$<$<AND:$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Debug>>:
		-Xcompiler ${MSVC_XCOMPILER_FLAGS} /Zi /Od>
		$<$<AND:$<CXX_COMPILER_ID:MSVC>,$<NOT:$<CONFIG:Debug>>>:
		-Xcompiler ${MSVC_XCOMPILER_FLAGS}>

		# Common CUDA flags
		-Xcudafe=--display_error_number
		--expt-extended-lambda
		--expt-relaxed-constexpr
		-Xptxas -warn-spills -res-usage
		--ptxas-options=-v

		# Per-config CUDA behavior
		$<$<CONFIG:Debug>:-G -lineinfo>              # full device debug, no heavy opt
		$<$<NOT:$<CONFIG:Debug>>:-lineinfo --use_fast_math> # fast math only outside Debug
)

add_library(developer_flags INTERFACE)
target_compile_options(developer_flags INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:${cxx_flags}>
    $<$<COMPILE_LANGUAGE:CUDA>:${cuda_flags}>
)
target_include_directories(developer_flags INTERFACE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

target_compile_features(developer_flags INTERFACE cxx_std_17)

#OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(developer_flags INTERFACE OpenMP::OpenMP_CXX)
endif()

#cudss
find_package(cudss)
message(STATUS "Found cuDSS version ${cudss_VERSION}")	
target_link_directories(developer_flags INTERFACE ${cudss_LIBRARY_DIR})

if (WIN32)
    target_include_directories(developer_flags INTERFACE ${cudss_INCLUDE_DIR})
    target_link_libraries(developer_flags INTERFACE cudss)
	
	#Copying cuDSS DLL to the binary directory
	add_custom_target(CopyCUDSSDLL)			
	foreach(CONFIG ${CMAKE_CONFIGURATION_TYPES})
		add_custom_command(
			OUTPUT "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CONFIG}/cudss64_0.dll"
			COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CONFIG}"
			COMMAND ${CMAKE_COMMAND} -E copy_if_different
				"${cudss_BINARY_DIR}/cudss64_0.dll"
				"${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CONFIG}/cudss64_0.dll"
			DEPENDS "${cudss_BINARY_DIR}/cudss64_0.dll"
			COMMENT "Copying cudss64_0.dll for ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CONFIG} configuration"
			VERBATIM
		)				
		add_custom_target("CopyCUDSSDLL${CONFIG}"
			DEPENDS "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CONFIG}/cudss64_0.dll")				
		add_dependencies(CopyCUDSSDLL "CopyCUDSSDLL${CONFIG}")				
	endforeach()
	
else()
    target_link_libraries(developer_flags INTERFACE cudss_static)
endif()


add_subdirectory(test_cudss)