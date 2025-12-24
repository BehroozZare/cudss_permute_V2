#include <assert.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <algorithm>
#include <cctype>
#include <string>
#include <vector>
#include <filesystem>

namespace fs = std::filesystem;

#include <cudss.h>

#include "matrix_market_reader.h"
#include "metis_permute.h"
#include "residual.h"
#include "util.h"
#include "csv_utils.h"

#define CUDA_SYNC_CHECK() do {                               \
    CUDA_ERROR(cudaGetLastError());                            \
    CUDA_ERROR(cudaDeviceSynchronize());                       \
  } while(0)


int main(int argc, char** argv)
{
    cudaDeviceProp p{};
    cudaGetDeviceProperties(&p, 0);
    printf("GPU: %s, cc=%d.%d\n", p.name, p.major, p.minor);

    if (argc < 2) {
        fprintf(stderr, "Usage: %s <matrix_file> [permute_type] [output_csv]\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char* matrix_filename = argv[1];

    std::string permute_type = "default";
    if (argc >= 3) {
        permute_type = argv[2];
        std::transform(permute_type.begin(),
                       permute_type.end(),
                       permute_type.begin(),
                       [](unsigned char c) { return std::tolower(c); });
    }

    std::string output_csv = "output";
    if (argc >= 4) {
        output_csv = argv[3];
    }


    cudssMatrixViewType_t mview = CUDSS_MVIEW_FULL;

    float ordering_time = 0.0f;
    float analysis_time = 0.0f;
    float factorization_time = 0.0f;
    float solve_time = 0.0f;

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

    // fill be with random numbers
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


    fprintf(stderr, "Created cuDSS library handle\n");
    // Creating the cuDSS library handle
    cudssHandle_t handle;
    CUDSS_ERROR(cudssCreate(&handle));

    // Creating cuDSS solver configuration and data objects
    cudssConfig_t solverConfig;
    cudssData_t   solverData;

    CUDSS_ERROR(cudssConfigCreate(&solverConfig));
    CUDSS_ERROR(cudssDataCreate(handle, &solverData));


    // set reorder type
    printf("\nUsing %s permuation", permute_type.c_str());
    if (permute_type == "symamd") {
        // an approximate minimum degree (AMD) reordering
        cudssAlgType_t reorder_alg = CUDSS_ALG_3;
        CUDSS_ERROR(cudssConfigSet(solverConfig,
                                   CUDSS_CONFIG_REORDERING_ALG,
                                   &reorder_alg,
                                   sizeof(cudssAlgType_t)));
    } else if (permute_type == "default") {
        // a customized nested dissection algorithm based on METIS
        cudssAlgType_t reorder_alg = CUDSS_ALG_DEFAULT;
        CUDSS_ERROR(cudssConfigSet(solverConfig,
                                   CUDSS_CONFIG_REORDERING_ALG,
                                   &reorder_alg,
                                   sizeof(cudssAlgType_t)));
    } else if (permute_type == "symrcm") {
        // a custom combination of block triangular reordering and
        // COLAMD
        cudssAlgType_t reorder_alg = CUDSS_ALG_1;
        CUDSS_ERROR(cudssConfigSet(solverConfig,
                                   CUDSS_CONFIG_REORDERING_ALG,
                                   &reorder_alg,
                                   sizeof(cudssAlgType_t)));
    } else if (permute_type == "metis") {
        idx_t* metis_perm = metis_permute(n, csr_offsets_h, csr_columns_h);
        CUDSS_ERROR(cudssDataSet(handle,
                                 solverData,
                                 CUDSS_DATA_USER_PERM,
                                 metis_perm,
                                 size_t(n * sizeof(int))));

    } else {
        fprintf(
            stderr, "\nInvalid permutation option %s", permute_type.c_str());
        return EXIT_FAILURE;
    }


    // Create matrix objects for the right-hand side b and solution x (as dense
    // matrices).
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


    // Create a matrix object for the sparse input matrix.
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
                                     CUDSS_MTYPE_SPD,  // CUDSS_MTYPE_SYMMETRIC,
                                     mview,
                                     CUDSS_BASE_ZERO));

    CUDATimer timer;

    float total_time = 0;


    auto check_cudss_info = [&](const char* tag) {
        int info = 0;
        size_t written = 0;
      
        // Many cuDSS samples treat INFO as an int status/index.
        // If this fails or size differs on your version, switch to the size-query method.
        cudssStatus_t st = cudssDataGet(handle, solverData,
                                       CUDSS_DATA_INFO,
                                       &info, sizeof(info), &written);
        if (st == CUDSS_STATUS_SUCCESS && written == sizeof(info) && info != 0) {
          fprintf(stderr, "\n[cudss] %s: CUDSS_DATA_INFO=%d\n", tag, info);
        }
      };


    // Permutation
    CUDA_SYNC_CHECK();
    timer.start();
    CUDSS_ERROR(cudssExecute(
        handle, CUDSS_PHASE_REORDERING, solverConfig, solverData, A, x, b));
    CUDA_SYNC_CHECK();
    timer.stop();
    ordering_time = timer.elapsed_millis();
    printf("\n cuDSS Permutation took: %f (ms)", ordering_time);
    CUDA_SYNC_CHECK();
    total_time += ordering_time;
    check_cudss_info("Reordering");
    // Symbolic factorization
    timer.start();
    CUDSS_ERROR(cudssExecute(handle,
                             CUDSS_PHASE_SYMBOLIC_FACTORIZATION,
                             solverConfig,
                             solverData,
                             A,
                             x,
                             b));
    CUDA_SYNC_CHECK();
    timer.stop();
    analysis_time = timer.elapsed_millis();
    check_cudss_info("Symbolic factorization");
    printf("\n cuDSS Symbolic factorization took: %f (ms)", analysis_time);
    total_time += analysis_time;
    //Saving the permutation and Elimination tree
    {
        size_t size_bytes = 0;
        size_t size_written = 0;
        // 1. Elimination Tree
        CUDSS_ERROR(cudssDataGet(handle, solverData, CUDSS_DATA_ELIMINATION_TREE, NULL, 0, &size_bytes));
        int* elim_tree = (int*)malloc(size_bytes);
        if (elim_tree) {
            CUDSS_ERROR(cudssDataGet(handle, solverData, CUDSS_DATA_ELIMINATION_TREE, elim_tree, size_bytes, &size_written));
            FILE* f_tree = fopen("/home/behrooz/Desktop/Last_Project/cudss_permute/output/elim_tree.txt", "w");
            if (f_tree) {
                int num_elements = size_written / sizeof(int);
                for (int i = 0; i < num_elements; ++i) {
                    fprintf(f_tree, "%d\n", elim_tree[i]);
                }
                fclose(f_tree);
                printf("\nSaved elimination tree to elim_tree.txt");
            }
            free(elim_tree);
        }

        // 2. Permutation (Using CUDSS_DATA_PERM_REORDER_ROW, as it seems to be the one for reordering permutation)
        size_bytes = 0;
        CUDSS_ERROR(cudssDataGet(handle, solverData, CUDSS_DATA_PERM_REORDER_ROW, NULL, 0, &size_bytes));
        int* perm = (int*)malloc(size_bytes);
        if (perm) {
            CUDSS_ERROR(cudssDataGet(handle, solverData, CUDSS_DATA_PERM_REORDER_ROW, perm, size_bytes, &size_written));
            FILE* f_perm = fopen("/home/behrooz/Desktop/Last_Project/cudss_permute/output/perm.txt", "w");
            if (f_perm) {
                int num_elements = size_written / sizeof(int);
                for (int i = 0; i < num_elements; ++i) {
                    fprintf(f_perm, "%d\n", perm[i]);
                }
                fclose(f_perm);
                printf("\nSaved permutation to perm.txt");
            }
            free(perm);
        }
    }
    // Factorization
    CUDA_SYNC_CHECK();
    timer.start();
    CUDSS_ERROR(cudssExecute(
        handle, CUDSS_PHASE_FACTORIZATION, solverConfig, solverData, A, x, b));
    CUDA_SYNC_CHECK();
    timer.stop();
    factorization_time = timer.elapsed_millis();
    check_cudss_info("Factorization");
    printf("\n cuDSS Factorization took: %f (ms)", factorization_time);
    total_time += factorization_time;

    // Solving
    CUDA_SYNC_CHECK();
    timer.start();
    CUDSS_ERROR(cudssExecute(
        handle, CUDSS_PHASE_SOLVE, solverConfig, solverData, A, x, b));
    CUDA_SYNC_CHECK();
    timer.stop();
    solve_time = timer.elapsed_millis();
    check_cudss_info("Solving");
    printf("\n cuDSS Solving took: %f (ms)", solve_time);
    total_time += solve_time;

    printf("\n\ncuDSS Total time: %f (ms)", total_time);

    // copy solution
    CUDA_ERROR(cudaMemcpy(
        x_values_h, x_values_d, n * sizeof(double), cudaMemcpyDeviceToHost));

    double residual = compute_residual_abs_norm(n,
                                                csr_offsets_h,
                                                csr_columns_h,
                                                csr_values_h,
                                                x_values_h,
                                                b_values_h,
                                                mview);

    printf("\nResidual L2 norm ||Ax - b|| = %e\n", residual);

    // Save to CSV
    {
        // Extract mesh name from path
        fs::path mesh_file(matrix_filename);
        std::string mesh_name = mesh_file.stem().string();

        // Create CSV with same headers as benchmark.cu
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

        CSVManager runtime_csv(output_csv, "some address", header, false);
        runtime_csv.addElementToRecord(mesh_name, "mesh_name");
        runtime_csv.addElementToRecord(permute_type, "ordering_type");
        runtime_csv.addElementToRecord(10, "nd_levels");
        runtime_csv.addElementToRecord(-1, "patch_type");
        runtime_csv.addElementToRecord(-1, "patch_size");
        runtime_csv.addElementToRecord(ordering_time, "ordering_time");
        runtime_csv.addElementToRecord(analysis_time, "analysis_time");
        runtime_csv.addElementToRecord(factorization_time, "factorization_time");
        runtime_csv.addElementToRecord(solve_time, "solve_time");
        runtime_csv.addElementToRecord(total_time, "total_time");
        runtime_csv.addElementToRecord(residual, "residual");
        runtime_csv.addRecord();
    }

    // free memory
    /* Destroying opaque objects, matrix wrappers and the cuDSS library handle
     */
    CUDSS_ERROR(cudssMatrixDestroy(A));
    CUDSS_ERROR(cudssMatrixDestroy(b));
    CUDSS_ERROR(cudssMatrixDestroy(x));
    CUDSS_ERROR(cudssDataDestroy(handle, solverData));
    CUDSS_ERROR(cudssConfigDestroy(solverConfig));
    CUDSS_ERROR(cudssDestroy(handle));


    if (csr_offsets_h) {
        free(csr_offsets_h);
    }
    if (csr_columns_h) {
        free(csr_columns_h);
    }
    if (csr_values_h) {
        free(csr_values_h);
    }
    if (x_values_h) {
        free(x_values_h);
    }
    if (b_values_h) {
        free(b_values_h);
    }
    CUDA_ERROR(cudaFree(csr_offsets_d));
    CUDA_ERROR(cudaFree(csr_columns_d));
    CUDA_ERROR(cudaFree(csr_values_d));
    CUDA_ERROR(cudaFree(x_values_d));
    CUDA_ERROR(cudaFree(b_values_d));

    return 0;
}
