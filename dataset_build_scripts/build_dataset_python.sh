#!/bin/bash
# Script to collect all merges on a small dataset

# Pass through all command line arguments to build_dataset.sh
dataset_build_scripts/build_dataset.sh input_data/repos_reaper_Python_0_1000.csv merges/repos_reaper_python "$@" --max_num_merges 9 --language python --test_size 1.0
