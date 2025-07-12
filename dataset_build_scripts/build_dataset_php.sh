#!/bin/bash
# Script to collect all merges on a PHP dataset

dataset_build_scripts/build_dataset.sh input_data/repos_reaper_PHP_0_1000.csv merges/repos_reaper_php "$@" --max_num_merges 50 --language php --test_size 1.0
