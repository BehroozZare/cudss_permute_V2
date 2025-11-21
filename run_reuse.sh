#!/bin/bash
# cmake --build build --target reuse
./build/bin/reuse data/nefertiti.mtx output/perm.txt output/elim_tree.txt

./build/bin/reuse data/nefertiti.mtx output_parth/perm_cudss.txt output_parth/elim_tree_cudss.txt

./build/bin/reuse data/nefertiti.mtx output_parth/perm_post_order.txt output_parth/elim_tree_post_order.txt

./build/bin/reuse data/nefertiti.mtx output_parth/perm_reverse.txt output_parth/elim_tree_reverse.txt
