#!/bin/bash
# Script to collect all merges on a C++ dataset

dataset_build_scripts/build_dataset.sh input_data/repos_reaper_C++_0_1000.csv merges/repos_reaper_cpp -g -m -b --max_num_merges 25 --language cpp --test_size 1.0
