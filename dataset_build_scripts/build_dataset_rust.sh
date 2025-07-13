#!/bin/bash
# Script to collect all merges on Rust dataset from GitHub

dataset_build_scripts/build_dataset.sh input_data/repos_github_Rust_0_1000.csv merges/repos_github_rust -g -m -b --max_num_merges 8 --language rust --test_size 1.0
