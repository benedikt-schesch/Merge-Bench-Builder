#!/bin/bash
# Script to collect all merges on Go dataset from GitHub

dataset_build_scripts/build_dataset.sh input_data/repos_github_Go_0_1000.csv merges/repos_github_go -g -m -b --max_num_merges 10 --language go --test_size 1.0
