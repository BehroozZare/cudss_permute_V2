#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <filesystem>

#include <parth/parth.h>
#include <CLI/CLI.hpp>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>


void get_cudss_mapping(int size, int max_level, std::vector<int>& cudss_mapping) {

    auto get_level = [&](int index) -> int {
        int level = 0;
        while(index != 0){
            index = (index - 1) / 2;
            level++;
        }
        return max_level - level;

    };
    cudss_mapping.clear();
    cudss_mapping.resize(size, -1);
    cudss_mapping[0] = 0;
    for(int i = 0; i < size; i++){
        int org_index = i;
        int level = get_level(org_index);
        assert(level <= max_level);
        int left_index = 2 * org_index + 1;
        int right_index = 2 * org_index + 2;
        int cudss_index = cudss_mapping[org_index];
        assert(cudss_index < size);
        assert(cudss_index != -1);
        int cudss_left_index = cudss_index + 1;
        int cudss_right_index = cudss_index + (1 << level);
        if (left_index < size) {
            cudss_mapping[left_index] = cudss_left_index;
        }
        if (right_index < size) {
            cudss_mapping[right_index] = cudss_right_index;
        }
    }
}


int post_order(int index, int offset, int size, std::vector<int>& post_order_mapping){
    int left_index = index * 2 + 1;
    int right_index = index * 2 + 2;
    int left_offset;
    int right_offset = offset;
    if(left_index < size){
        right_offset = post_order(left_index, offset, size, post_order_mapping);
    }
    int current_offset = right_offset;
    if(right_index < size){
        current_offset = post_order(right_index, current_offset, size, post_order_mapping);
    }
    post_order_mapping[index] = current_offset;
    offset = current_offset + 1;
    return offset;
}

void get_post_order_mapping(int size, std::vector<int>& post_order_mapping) {
    post_order_mapping.clear();
    post_order_mapping.resize(size, 0);
    post_order(0, 0, size, post_order_mapping);
}


void get_reverse_mapping(int size, std::vector<int>& reverse_mapping) {
    reverse_mapping.clear();
    reverse_mapping.resize(size, 0);
    for(int i = 0; i < size; i++){
        reverse_mapping[size - 1 - i] = i;
    }
}

void apply_mapping(const std::vector<int>& mapping, PARTH::ParthAPI& parth, std::vector<int>& etree) {
    etree.clear();
    etree.resize(mapping.size(), 0);
    for (int i = 0; i < mapping.size(); i++) {
        int e_tree_index = mapping[i];
        int e_tree_value = parth.hmd.HMD_tree[i].DOFs.size();
        etree[e_tree_index] = e_tree_value;
    }
}

void save_elimination_tree(const std::vector<int>& elim_tree, const std::string& filename) {
    std::ofstream out(filename);
    if (!out.is_open()) {
        std::cerr << "Failed to open " << filename << " for writing." << std::endl;
        return;
    }
    std::cout << "Saving elimination tree with " << elim_tree.size() << " elements." << std::endl;
    for (int i = 0; i < elim_tree.size(); i++) {
        out << elim_tree[i] << "\n";
    }
    out.close();
    std::cout << "Saved elimination tree to " << filename << std::endl;
}

// Function to save permutation to file
void save_permutation(const std::vector<int>& perm, const std::string& filename) {
    std::ofstream out(filename);
    if (!out.is_open()) {
        std::cerr << "Failed to open " << filename << " for writing." << std::endl;
        return;
    }
    std::cout << "Saving permutation with " << perm.size() << " elements." << std::endl;
    for (int p : perm) {
        out << p << "\n";
    }
    out.close();
    std::cout << "Saved permutation to " << filename << std::endl;
}

bool test_etree_correctness(const std::vector<int>& etree, const std::vector<int>& perm) {
    int total_size = 0;
    for(auto& e : etree){
        total_size += e;
    }
    if(total_size != perm.size()){
        return false;
    }
    return true;
}

int main(int argc, char *argv[]) {
    CLI::App app{"Parth Integration Example"};

    std::string original_matrix_path;
    std::string modified_matrix_path;
    std::string output_dir = "output_parth";

    app.add_option("-o,--original", original_matrix_path, "Path to the original matrix (.mtx)")->required();
    app.add_option("-m,--modified", modified_matrix_path, "Path to the modified matrix (.mtx)");
    app.add_option("--output-dir", output_dir, "Directory to save outputs");

    CLI11_PARSE(app, argc, argv);

    // Ensure output directory exists
    std::filesystem::create_directories(output_dir);

    // =======================================================================
    // STEP 1: Load the Original Matrix
    // =======================================================================
    std::cout << "=== STEP 1: Loading Original Matrix ===" << std::endl;
    std::cout << "Loading original matrix from: " << original_matrix_path << std::endl;
    Eigen::SparseMatrix<double> original_matrix;
    if (!Eigen::loadMarket(original_matrix, original_matrix_path)) {
        std::cerr << "Failed to load original matrix from: " << original_matrix_path << std::endl;
        return 1;
    }
    std::cout << "Original matrix loaded successfully. Size: " << original_matrix.rows() << "x" << original_matrix.cols() 
              << ", Non-zeros: " << original_matrix.nonZeros() << std::endl;

    // =======================================================================
    // STEP 2: Computing Permutation for Original Matrix
    // =======================================================================
    std::cout << "\n=== STEP 2: Computing Permutation for Original Matrix ===" << std::endl;
    std::cout << "Initializing PARTH API..." << std::endl;
    
    PARTH::ParthAPI parth;
    parth.setNDLevels(9);
    
    std::cout << "Setting original matrix data in PARTH..." << std::endl;
    parth.setMatrix(original_matrix.rows(),
                    const_cast<int*>(original_matrix.outerIndexPtr()), 
                    const_cast<int*>(original_matrix.innerIndexPtr()), 1);
    
    std::cout << "Computing permutation (from scratch)..." << std::endl;
    std::vector<int> perm;
    parth.computePermutation(perm, 1);
    
    std::cout << "=== TIMING FOR ORIGINAL MATRIX (Cold Start) ===" << std::endl;
    parth.printTiming();
    
    // Save permutation for original matrix
    save_permutation(perm, output_dir + "/perm_original.txt");

    std::cout << "Saving elimination tree for original matrix..." << std::endl;
    std::cout << "HMD tree size: " << parth.hmd.HMD_tree.size() << std::endl;
    std::vector<int> etree(parth.hmd.HMD_tree.size(), 0);

    //Apply CUDSS mapping
    std::vector<int> cudss_mapping(parth.hmd.HMD_tree.size(), 0);
    get_cudss_mapping(parth.hmd.HMD_tree.size(), parth.getNDLevels(), cudss_mapping);
    apply_mapping(cudss_mapping, parth, etree);
    save_elimination_tree(etree, output_dir + "/elim_tree_cudss.txt");
    if(!test_etree_correctness(etree, perm)){
        std::cerr << "CUDSS mapping is incorrect" << std::endl;
        return 1;
    }

    //Apply post order mapping
    std::vector<int> post_order_mapping(parth.hmd.HMD_tree.size(), 0);
    get_post_order_mapping(parth.hmd.HMD_tree.size(), post_order_mapping);
    apply_mapping(post_order_mapping, parth, etree);
    save_elimination_tree(etree, output_dir + "/elim_tree_post_order.txt");
    if(!test_etree_correctness(etree, perm)){
        std::cerr << "Post order mapping is incorrect" << std::endl;
        return 1;
    }
    //Apply reverse mapping
    std::vector<int> reverse_mapping(parth.hmd.HMD_tree.size(), 0);
    get_reverse_mapping(parth.hmd.HMD_tree.size(), reverse_mapping);
    apply_mapping(reverse_mapping, parth, etree);
    save_elimination_tree(etree, output_dir + "/elim_tree_reverse.txt");
    if(!test_etree_correctness(etree, perm)){
        std::cerr << "Reverse mapping is incorrect" << std::endl;
        return 1;
    }
    return 0;
}

