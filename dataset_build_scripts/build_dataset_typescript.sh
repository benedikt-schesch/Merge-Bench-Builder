#!/bin/bash
# Script to collect all merges on TypeScript dataset from GitHub

dataset_build_scripts/build_dataset.sh input_data/repos_github_TypeScript_0_1200.csv merges/repos_github_typescript "$@" --max_num_merges 100 --language typescript --test_size 1.0
