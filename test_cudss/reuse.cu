#include <assert.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <algorithm>
#include <cctype>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <cstdlib>

#include <cudss.h>

#include "matrix_market_reader.h"
#include "metis_permute.h"
#include "residual.h"
#include "util.h"

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

int main(int argc, char** argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <matrix_file>\n", argv[0]);
        return EXIT_FAILURE;
    }
    const char* matrix_filename = argv[1];

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

    // NOTE: paths as per your generate.cu setup
    const char* perm_path = "/home/behrooz/Desktop/Last_Project/cudss_permute/output/perm.txt";
    const char* tree_path = "/home/behrooz/Desktop/Last_Project/cudss_permute/output/elim_tree.txt";

    int perm_size   = 0;
    int elim_size   = 0;
    int* user_perm  = nullptr;
    int* user_etree = nullptr;

    // --- permutation (must have size n) --------------------------------------
    user_perm = read_int_array(perm_path, perm_size);
    if (!user_perm) {
        fprintf(stderr, "Error loading permutation from %s\n", perm_path);
        return EXIT_FAILURE;
    }
    if (perm_size != n) {
        fprintf(stderr,
                "Error: permutation file %s has %d entries, but matrix has n=%d\n",
                perm_path, perm_size, n);
        free(user_perm);
        return EXIT_FAILURE;
    }

    printf("Loaded permutation of size %d from %s\n", perm_size, perm_path);

    // Set user permutation (cuDSS copies it internally, per docs)
    CUDSS_ERROR(cudssDataSet(handle,
                             solverData,
                             CUDSS_DATA_USER_PERM,
                             user_perm,
                             size_t(perm_size * sizeof(int))));

    // --- elimination tree (size is NOT n; we just trust the file) -----------
    user_etree = read_int_array(tree_path, elim_size);
    if (!user_etree) {
        fprintf(stderr,
                "Warning: could not load elimination tree from %s; "
                "cuDSS will recompute it.\n",
                tree_path);
    } else {
        printf("Loaded elimination tree of size %d from %s\n",
               elim_size, tree_path);

        // Pass user elimination tree to cuDSS. It **must** be used together
        // with CUDSS_DATA_USER_PERM to have effect.
        CUDSS_ERROR(cudssDataSet(handle,
                                 solverData,
                                 CUDSS_DATA_USER_ELIMINATION_TREE,
                                 user_etree,
                                 size_t(elim_size * sizeof(int))));
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
    total_time += timer.elapsed_millis();

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
    total_time += timer.elapsed_millis();

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
    total_time += timer.elapsed_millis();

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
    total_time += timer.elapsed_millis();

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

    return 0;
}
