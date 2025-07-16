#!/bin/bash
# Script to collect all merges on a C dataset

# Pass through all command line arguments to build_dataset.sh
dataset_build_scripts/build_dataset.sh input_data/repos_reaper_C_0_1000.csv merges/repos_reaper_c "$@" --max_num_merges 6 --language c --test_size 1.0
