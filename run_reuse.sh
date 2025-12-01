#!/bin/bash
# cmake --build build --target reuse
./build/bin/reuse data/nefertiti.mtx output/perm.txt output/elim_tree.txt

./build/bin/reuse data/nefertiti.mtx output_parth/user_defined_perm.txt output_parth/user_defined_etree.txt
