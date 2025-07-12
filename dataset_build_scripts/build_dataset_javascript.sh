#!/bin/bash
# Script to collect all merges on JavaScript dataset from GitHub

dataset_build_scripts/build_dataset.sh input_data/repos_github_JavaScript_0_1200.csv merges/repos_github_javascript -g -m -b --max_num_merges 25 --language javascript --test_size 1.0
