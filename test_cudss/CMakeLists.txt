#CMake minimum requirement 
cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

set(exec_name "test_cudss")

# Source files common to both executables
set(common_sources util.h matrix_market_reader.h residual.h metis_permute.h)

# Generate executable
add_executable(generate generate.cu ${common_sources})
target_link_libraries(generate
	$<BUILD_INTERFACE:developer_flags>
	metis
)

# Reuse executable
add_executable(reuse reuse.cu ${common_sources})
target_link_libraries(reuse
	$<BUILD_INTERFACE:developer_flags>
	metis
)

# Generate with Parth executable
add_executable(generate_with_parth generate_with_parth.cu)
target_link_libraries(generate_with_parth
    $<BUILD_INTERFACE:developer_flags>
    parth
    Eigen3::Eigen
    CLI11::CLI11
)
