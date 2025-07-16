#!/bin/bash
# Script to collect all merges on Go dataset from GitHub

# Pass through all command line arguments to build_dataset.sh
dataset_build_scripts/build_dataset.sh input_data/repos_github_Go_0_1000.csv merges/repos_github_go "$@" --max_num_merges 7 --language go --test_size 1.0
