#!/bin/bash
# cmake --build build --target reuse

#Extract all the mesh names from the benchmark folder (those finished *.mtx)
base_folder="/media/behrooz/FarazHard/Checkpoints"
mesh_names=$(ls $base_folder/*.mtx | xargs -n1 basename | sed 's/\.mtx//g')
output_file="output"
for mesh_name in $mesh_names; do
    echo "Generating and benchmarking $base_folder/$mesh_name"
    ./build/bin/generate $base_folder/$mesh_name.mtx default $output_file
    ./build/bin/benchmark $base_folder/$mesh_name.mtx $output_file
done
