#include <assert.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <algorithm>
#include <cctype>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <filesystem>

namespace fs = std::filesystem;

#include <cudss.h>

#include "matrix_market_reader.h"
#include "metis_permute.h"
#include "residual.h"
#include "util.h"
#include "csv_utils.h"


// -----------------------------------------------------------------------------
// Helper to read an int array from a text file.
// Each line (or whitespace separated token) is parsed as an int.
// Returns a malloc'ed array and writes its logical size to out_size.
// Caller owns the returned pointer and must free() it.
// -----------------------------------------------------------------------------
int* read_int_array(const char* filename, int& out_size)
{
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        fprintf(stderr, "Failed to open %s\n", filename);
        out_size = 0;
        return nullptr;
    }

    std::vector<int> data;
    int val;
    while (infile >> val) {
        data.push_back(val);
    }
    infile.close();

    out_size = static_cast<int>(data.size());
    if (out_size == 0) {
        fprintf(stderr, "Warning: %s is empty or could not be parsed\n", filename);
        return nullptr;
    }

    int* arr = (int*)malloc(out_size * sizeof(int));
    if (!arr) {
        fprintf(stderr, "malloc failed for %s (size %d)\n", filename, out_size);
        out_size = 0;
        return nullptr;
    }

    std::copy(data.begin(), data.end(), arr);
    return arr;
}


struct BenchmarkParameters {
    std::string base_folder;
    std::string mesh_name_csv = "mesh_name";
    std::string ordering_type_csv = "ordering_type";
    std::string patch_type_csv = "patch_type";
    std::string patch_size_csv = "patch_size";
    std::string nd_levels_csv = "nd_levels";

    std::string mesh_name;
    std::string ordering_type;
    int nd_levels;
    int patch_size;
    std::string patch_type;
    float ordering_time_from_file;

    std::string perm_path;
    std::string tree_path;

    void read_parameters(const std::string& perm_filename) {
        // Store the perm filename (just the filename, not full path)
        this->perm_path = perm_filename;
        
        // Build tree_path by replacing "perm_" with "etree_"
        this->tree_path = perm_filename;
        size_t pos = this->tree_path.find("perm_");
        if (pos != std::string::npos) {
            this->tree_path.replace(pos, 5, "etree_");
        }

        // Parse filename: perm_<mesh>_<ordering>_level=X,patch_type=Y,patch_size=Z,ordering_time=W.txt
        // Remove "perm_" prefix and ".txt" suffix
        std::string name = perm_filename;
        if (name.rfind("perm_", 0) == 0) {
            name = name.substr(5);  // Remove "perm_"
        }
        if (name.size() > 4 && name.substr(name.size() - 4) == ".txt") {
            name = name.substr(0, name.size() - 4);  // Remove ".txt"
        }

        // Find "_level=" to split mesh_name_ordering_type from parameters
        size_t level_pos = name.find("_level=");
        if (level_pos == std::string::npos) {
            fprintf(stderr, "Warning: Could not parse filename %s\n", perm_filename.c_str());
            return;
        }

        std::string mesh_and_ordering = name.substr(0, level_pos);
        std::string params_str = name.substr(level_pos + 1);  // "level=X,patch_type=Y,..."

        // Split mesh_and_ordering: find last underscore before ordering type
        // Format: <mesh_name>_<ordering_type> where ordering_type is like PARTH or PATCH_ORDERING
        // We need to find where mesh_name ends and ordering_type begins
        // ordering_type can contain underscores (e.g., PATCH_ORDERING)
        // Strategy: look for known ordering types or find the pattern
        
        // Look for known ordering types
        size_t ordering_start = std::string::npos;
        if (mesh_and_ordering.find("_PARTH") != std::string::npos) {
            ordering_start = mesh_and_ordering.find("_PARTH");
            this->ordering_type = mesh_and_ordering.substr(ordering_start + 1);
            this->mesh_name = mesh_and_ordering.substr(0, ordering_start);
        } else if (mesh_and_ordering.find("_PATCH_ORDERING") != std::string::npos) {
            ordering_start = mesh_and_ordering.find("_PATCH_ORDERING");
            this->ordering_type = mesh_and_ordering.substr(ordering_start + 1);
            this->mesh_name = mesh_and_ordering.substr(0, ordering_start);
        } else {
            // Fallback: assume last underscore separates mesh from ordering
            size_t last_underscore = mesh_and_ordering.rfind('_');
            if (last_underscore != std::string::npos) {
                this->mesh_name = mesh_and_ordering.substr(0, last_underscore);
                this->ordering_type = mesh_and_ordering.substr(last_underscore + 1);
            } else {
                this->mesh_name = mesh_and_ordering;
                this->ordering_type = "UNKNOWN";
            }
        }

        // Parse key=value pairs from params_str (comma-separated)
        // Format: level=X,patch_type=Y,patch_size=Z,ordering_time=W
        std::string token;
        std::istringstream token_stream(params_str);
        while (std::getline(token_stream, token, ',')) {
            size_t eq_pos = token.find('=');
            if (eq_pos == std::string::npos) continue;
            
            std::string key = token.substr(0, eq_pos);
            std::string value = token.substr(eq_pos + 1);

            if (key == "level") {
                this->nd_levels = std::stoi(value);
            } else if (key == "patch_type") {
                this->patch_type = value;
            } else if (key == "patch_size") {
                this->patch_size = std::stoi(value);
            } else if (key == "ordering_time") {
                this->ordering_time_from_file = std::stof(value);
            }
        }
    }
};



void prepare_benchmark(const std::string& mesh_path, std::vector<BenchmarkParameters>& benchmark_parameters) {
    // Extract folder and mesh name from the mesh path
    fs::path mesh_file(mesh_path);
    std::string benchmark_folder = mesh_file.parent_path().string();
    std::string mesh_name = mesh_file.stem().string();  // e.g., "fish" from "fish.mtx"
    
    // Build the prefix to match: "perm_<mesh_name>_"
    std::string perm_prefix = "perm_" + mesh_name + "_";
    
    printf("Looking for benchmark files matching: %s*.txt\n", perm_prefix.c_str());
    
    // Iterate over all files in the benchmark folder
    for (const auto& entry : fs::directory_iterator(benchmark_folder)) {
        if (!entry.is_regular_file()) continue;
        
        std::string filename = entry.path().filename().string();
        
        // Only process perm_<mesh_name>_*.txt files
        if (filename.rfind(perm_prefix, 0) == 0 && 
            filename.size() > 4 && 
            filename.substr(filename.size() - 4) == ".txt") {
            
            BenchmarkParameters params;
            params.base_folder = benchmark_folder;
            params.read_parameters(filename);
            benchmark_parameters.push_back(params);
        }
    }
    
    printf("Found %zu benchmark configurations for mesh '%s'\n", 
           benchmark_parameters.size(), mesh_name.c_str());
}

int main(int argc, char** argv)
{
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <benchmark_folder> <output_file> \n", argv[0]);
        return EXIT_FAILURE;
    }
    
    std::string mesh_address = std::string(argv[1]);
    std::string output_file = std::string(argv[2]);
    std::vector<BenchmarkParameters> benchmark_parameters;
    prepare_benchmark(mesh_address, benchmark_parameters);


    for(const auto& parameters : benchmark_parameters) {

        std::string matrix_filename = parameters.base_folder + "/" + parameters.mesh_name + ".mtx";
        std::string perm_path = parameters.base_folder + "/" + parameters.perm_path;
        std::string tree_path = parameters.base_folder + "/" + parameters.tree_path;

        float ordering_time = 0.0f;
        float analysis_time = 0.0f;
        float factorization_time = 0.0f;
        float solve_time = 0.0f;

        cudssMatrixViewType_t mview = CUDSS_MVIEW_FULL;

        int n;
        int nnz;

        int*    csr_offsets_h = NULL;
        int*    csr_columns_h = NULL;
        double* csr_values_h  = NULL;
        double *x_values_h = NULL, *b_values_h = NULL;

        int*    csr_offsets_d = NULL;
        int*    csr_columns_d = NULL;
        double* csr_values_d  = NULL;
        double *x_values_d = NULL, *b_values_d = NULL;

        int failed = matrix_reader(matrix_filename,
                                n,
                                nnz,
                                &csr_offsets_h,
                                &csr_columns_h,
                                &csr_values_h,
                                mview);
        if (failed) {
            fprintf(stderr, "Reader failed.\n");
            return EXIT_FAILURE;
        }

        printf("solving a real linear %dx%d system from file \"%s\"\n",
            n,
            n,
            matrix_filename);

        // Allocate host memory x and b
        x_values_h = (double*)malloc(n * sizeof(double));
        b_values_h = (double*)malloc(n * sizeof(double));

        if (!csr_offsets_h || !csr_columns_h || !csr_values_h || !x_values_h ||
            !b_values_h) {
            fprintf(stderr, "Error: host memory allocation failed\n");
            return EXIT_FAILURE;
        }

        // fill b with random numbers
        fill_random(n, b_values_h);

        // allocate device memory for A, x and b
        CUDA_ERROR(cudaMalloc(&csr_offsets_d, (n + 1) * sizeof(int)));
        CUDA_ERROR(cudaMalloc(&csr_columns_d, nnz * sizeof(int)));
        CUDA_ERROR(cudaMalloc(&csr_values_d, nnz * sizeof(double)));
        CUDA_ERROR(cudaMalloc(&b_values_d, n * sizeof(double)));
        CUDA_ERROR(cudaMalloc(&x_values_d, n * sizeof(double)));

        // move memory from host to device
        CUDA_ERROR(cudaMemcpy(csr_offsets_d,
                            csr_offsets_h,
                            (n + 1) * sizeof(int),
                            cudaMemcpyHostToDevice));
        CUDA_ERROR(cudaMemcpy(csr_columns_d,
                            csr_columns_h,
                            nnz * sizeof(int),
                            cudaMemcpyHostToDevice));
        CUDA_ERROR(cudaMemcpy(csr_values_d,
                            csr_values_h,
                            nnz * sizeof(double),
                            cudaMemcpyHostToDevice));
        CUDA_ERROR(cudaMemcpy(
            b_values_d, b_values_h, n * sizeof(double), cudaMemcpyHostToDevice));

        // Creating the cuDSS library handle
        cudssHandle_t handle;
        CUDSS_ERROR(cudssCreate(&handle));

        // Creating cuDSS solver configuration and data objects
        cudssConfig_t solverConfig;
        cudssData_t   solverData;

        CUDSS_ERROR(cudssConfigCreate(&solverConfig));
        CUDSS_ERROR(cudssDataCreate(handle, &solverData));

        // -------------------------------------------------------------------------
        // LOAD PERMUTATION AND ELIMINATION TREE AND PASS THEM TO cuDSS
        // -------------------------------------------------------------------------
        printf("\nLoading permutation and elimination tree from files...\n");

        // NOTE: paths passed as arguments

        int perm_size   = 0;
        int elim_size   = 0;
        int* user_perm  = nullptr;
        int* user_etree = nullptr;

        // --- permutation (must have size n) --------------------------------------
        user_perm = read_int_array(perm_path.c_str(), perm_size);
        if (!user_perm) {
            fprintf(stderr, "Error loading permutation from %s\n", perm_path.c_str());
            return EXIT_FAILURE;
        }
        if (perm_size != n) {
            fprintf(stderr,
                    "Error: permutation file %s has %d entries, but matrix has n=%d\n",
                    perm_path.c_str(), perm_size, n);
            free(user_perm);
            return EXIT_FAILURE;
        }

        printf("Loaded permutation of size %d from %s\n", perm_size, perm_path.c_str());

        // Set user permutation (cuDSS copies it internally, per docs)
        CUDSS_ERROR(cudssDataSet(handle,
                                solverData,
                                CUDSS_DATA_USER_PERM,
                                user_perm,
                                size_t(perm_size * sizeof(int))));

        // --- elimination tree (size is NOT n; we just trust the file) -----------
        user_etree = read_int_array(tree_path.c_str(), elim_size);
        if (!user_etree) {
            fprintf(stderr,
                    "Warning: could not load elimination tree from %s; "
                    "cuDSS will recompute it.\n",
                    tree_path.c_str());
        } else {
            printf("Loaded elimination tree of size %d from %s\n",
                elim_size, tree_path.c_str());

            // Pass user elimination tree to cuDSS. It **must** be used together
            // with CUDSS_DATA_USER_PERM to have effect.
            CUDSS_ERROR(cudssDataSet(handle,
                                    solverData,
                                    CUDSS_DATA_USER_ELIMINATION_TREE,
                                    user_etree,
                                    size_t(elim_size * sizeof(int))));

            //Set the etree levels
            int level = std::log2(elim_size + 1);
            CUDSS_ERROR(cudssConfigSet(solverConfig,
                 CUDSS_CONFIG_ND_NLEVELS, &level, sizeof(int)));
            printf("Set the number of levels in the etree to %d\n", level);
        }
        // From docs: both USER_PERM and USER_ELIMINATION_TREE are copied into
        // internal buffers, so we *could* free them now. To keep it simple and
        // super-safe, we free them at the very end of main.

        // -------------------------------------------------------------------------
        // Create matrix objects for RHS b and solution x (dense)
        // -------------------------------------------------------------------------
        cudssMatrix_t x, b;

        int64_t nrows = n, ncols = n;
        int     ldb = ncols, ldx = nrows;
        CUDSS_ERROR(cudssMatrixCreateDn(&b,
                                        ncols,
                                        1,
                                        ldb,
                                        b_values_d,
                                        cuda_type<double>(),
                                        CUDSS_LAYOUT_COL_MAJOR));

        CUDSS_ERROR(cudssMatrixCreateDn(&x,
                                        nrows,
                                        1,
                                        ldx,
                                        x_values_d,
                                        cuda_type<double>(),
                                        CUDSS_LAYOUT_COL_MAJOR));

        // Create matrix object for sparse input matrix
        cudssMatrix_t A;
        CUDSS_ERROR(cudssMatrixCreateCsr(&A,
                                        nrows,
                                        ncols,
                                        nnz,
                                        csr_offsets_d,
                                        NULL,
                                        csr_columns_d,
                                        csr_values_d,
                                        CUDA_R_32I,
                                        cuda_type<double>(),
                                        CUDSS_MTYPE_SPD,
                                        mview,
                                        CUDSS_BASE_ZERO));

        CUDATimer timer;
        float total_time = 0.0f;

        // Reordering Phase (now uses user perm + user elimination tree)
        printf("\nExecuting Reordering Phase (using user perm + elim tree)...\n");
        timer.start();
        CUDSS_ERROR(cudssExecute(handle,
                                CUDSS_PHASE_REORDERING,
                                solverConfig,
                                solverData,
                                A,
                                x,
                                b));
        timer.stop();
        printf(" cuDSS Reordering took: %f (ms)\n", timer.elapsed_millis());
        ordering_time += timer.elapsed_millis();

        // Symbolic factorization
        timer.start();
        CUDSS_ERROR(cudssExecute(handle,
                                CUDSS_PHASE_SYMBOLIC_FACTORIZATION,
                                solverConfig,
                                solverData,
                                A,
                                x,
                                b));
        timer.stop();
        printf(" cuDSS Symbolic factorization took: %f (ms)\n", timer.elapsed_millis());
        analysis_time += timer.elapsed_millis();

        // Factorization
        timer.start();
        CUDSS_ERROR(cudssExecute(handle,
                                CUDSS_PHASE_FACTORIZATION,
                                solverConfig,
                                solverData,
                                A,
                                x,
                                b));
        timer.stop();
        printf(" cuDSS Factorization took: %f (ms)\n", timer.elapsed_millis());
        factorization_time += timer.elapsed_millis();

        // Solving
        timer.start();
        CUDSS_ERROR(cudssExecute(handle,
                                CUDSS_PHASE_SOLVE,
                                solverConfig,
                                solverData,
                                A,
                                x,
                                b));
        timer.stop();
        printf(" cuDSS Solving took: %f (ms)\n", timer.elapsed_millis());
        solve_time += timer.elapsed_millis();

        total_time = parameters.ordering_time_from_file + ordering_time + analysis_time + factorization_time + solve_time;
        printf("\n\ncuDSS Total time: %f (ms)\n", total_time);

        // copy solution back
        CUDA_ERROR(cudaMemcpy(
            x_values_h, x_values_d, n * sizeof(double), cudaMemcpyDeviceToHost));

        double residual = compute_residual_abs_norm(n,
                                                    csr_offsets_h,
                                                    csr_columns_h,
                                                    csr_values_h,
                                                    x_values_h,
                                                    b_values_h,
                                                    mview);

        printf("Residual L2 norm ||Ax - b|| = %e\n", residual);

        // -------------------------------------------------------------------------
        // Cleanup
        // -------------------------------------------------------------------------
        CUDSS_ERROR(cudssMatrixDestroy(A));
        CUDSS_ERROR(cudssMatrixDestroy(b));
        CUDSS_ERROR(cudssMatrixDestroy(x));
        CUDSS_ERROR(cudssDataDestroy(handle, solverData));
        CUDSS_ERROR(cudssConfigDestroy(solverConfig));
        CUDSS_ERROR(cudssDestroy(handle));

        if (user_perm)  free(user_perm);
        if (user_etree) free(user_etree);

        if (csr_offsets_h) free(csr_offsets_h);
        if (csr_columns_h) free(csr_columns_h);
        if (csr_values_h) free(csr_values_h);
        if (x_values_h)   free(x_values_h);
        if (b_values_h)   free(b_values_h);

        CUDA_ERROR(cudaFree(csr_offsets_d));
        CUDA_ERROR(cudaFree(csr_columns_d));
        CUDA_ERROR(cudaFree(csr_values_d));
        CUDA_ERROR(cudaFree(x_values_d));
        CUDA_ERROR(cudaFree(b_values_d));


        std::string csv_name = output_file;
        std::string mesh_name = parameters.mesh_name;
        std::vector<std::string> header;
        header.emplace_back("mesh_name");
        header.emplace_back("ordering_type");
        header.emplace_back("nd_levels");
        header.emplace_back("patch_type");
        header.emplace_back("patch_size");
        header.emplace_back("ordering_time");
        header.emplace_back("analysis_time");
        header.emplace_back("factorization_time");
        header.emplace_back("solve_time");
        header.emplace_back("total_time");
        header.emplace_back("residual");

        CSVManager runtime_csv(csv_name, "some address", header, false);
        runtime_csv.addElementToRecord(mesh_name, "mesh_name");
        runtime_csv.addElementToRecord(parameters.ordering_type, "ordering_type");
        runtime_csv.addElementToRecord(parameters.nd_levels, "nd_levels");
        runtime_csv.addElementToRecord(parameters.patch_type, "patch_type");
        runtime_csv.addElementToRecord(parameters.patch_size, "patch_size");
        runtime_csv.addElementToRecord(ordering_time + parameters.ordering_time_from_file, "ordering_time");
        runtime_csv.addElementToRecord(analysis_time, "analysis_time");
        runtime_csv.addElementToRecord(factorization_time, "factorization_time");
        runtime_csv.addElementToRecord(solve_time, "solve_time");
        runtime_csv.addElementToRecord(total_time, "total_time");
        runtime_csv.addElementToRecord(residual, "residual");
        runtime_csv.addRecord();


    }

    return 0;
}
