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



void assemble_perm(std::vector<int>& etree_map, PARTH::ParthAPI& parth, std::vector<int>& perm) {
    perm.clear();
    perm.resize(parth.M_n, -1);
    std::vector<int> etree_inverse(etree_map.size(), 0);
    for(int i = 0; i < etree_map.size(); i++){
        etree_inverse[etree_map[i]] = i;
    }

    int offset = 0;
    for(int i = 0; i < etree_inverse.size(); i++){
        int etree_id = etree_inverse[i];
        auto& node = parth.hmd.HMD_tree[etree_id];
        if (node.DOFs.empty())
            continue;
        for (int local_node = 0; local_node < node.DOFs.size(); local_node++) {
            int global_node = node.DOFs[local_node];
            int perm_index  = node.permuted_new_label[local_node] + offset;
            assert(global_node >= 0 && global_node < parth.M_n &&
                    "Invalid global node index");
            assert(perm_index >= 0 && perm_index < perm.size() &&
                    "Permutation index out of bounds");
            assert(perm[perm_index] == -1 &&
                    "Permutation slot already filled - duplicate node!");
            perm[perm_index] = global_node;
        }
        offset += node.DOFs.size();
    }
}



void level_numbering(int size, std::vector<int>& hmd_to_etree) {
    hmd_to_etree.clear();
    hmd_to_etree.resize(size, 0);
    for(int i = 0; i < size; i++){
        hmd_to_etree[size - 1 - i] = i;
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


bool test_perm_correctness(const std::vector<int>& perm) {
    for(int i = 0; i < perm.size(); i++){
        if(perm[i] == -1){
            return false;
        }
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
    
    //Apply reverse mapping
    std::vector<int> hmd_to_etree(parth.hmd.HMD_tree.size(), 0);
    std::vector<int> cudss_perm(parth.M_n, -1);
    std::vector<int> etree;
    level_numbering(parth.hmd.HMD_tree.size(), hmd_to_etree);
    apply_mapping(hmd_to_etree, parth, etree);
    assemble_perm(hmd_to_etree, parth, cudss_perm);
    save_elimination_tree(etree, output_dir + "/user_defined_etree.txt");
    save_permutation(cudss_perm, output_dir + "/user_defined_perm.txt");
    if(!test_etree_correctness(etree, cudss_perm)){
        std::cerr << "User defined mapping is incorrect" << std::endl;
        return 1;
    }
    return 0;
}

