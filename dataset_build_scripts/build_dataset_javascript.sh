#!/bin/bash
# Script to collect all merges on JavaScript dataset from GitHub

# Pass through all command line arguments to build_dataset.sh
dataset_build_scripts/build_dataset.sh input_data/repos_github_Javascript_0_1000.csv merges/repos_github_javascript "$@" --max_num_merges 12 --language javascript --test_size 1.0
