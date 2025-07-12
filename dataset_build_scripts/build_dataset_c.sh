#!/bin/bash
# Script to collect all merges on a C dataset

dataset_build_scripts/build_dataset.sh input_data/repos_reaper_C_0_1000.csv merges/repos_reaper_c -g -m -b --max_num_merges 6 --language c --test_size 1.0
