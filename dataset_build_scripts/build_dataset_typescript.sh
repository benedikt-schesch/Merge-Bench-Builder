#!/bin/bash
# Script to collect all merges on TypeScript dataset from GitHub

dataset_build_scripts/build_dataset.sh input_data/repos_github_Typescript_0_1000.csv merges/repos_github_typescript -g -m -b --max_num_merges 10 --language typescript --test_size 1.0
