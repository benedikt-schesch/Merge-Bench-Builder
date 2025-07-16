#!/bin/bash
# Script to collect all merges on TypeScript dataset from GitHub

# Pass through all command line arguments to build_dataset.sh
dataset_build_scripts/build_dataset.sh input_data/repos_github_Typescript_0_1000.csv merges/repos_github_typescript "$@" --max_num_merges 10 --language typescript --test_size 1.0
