#!/bin/bash
# Script to collect all merges on a PHP dataset

# Pass through all command line arguments to build_dataset.sh
dataset_build_scripts/build_dataset.sh input_data/repos_reaper_PHP_0_1000.csv merges/repos_reaper_php "$@" --max_num_merges 11 --language php --test_size 1.0
