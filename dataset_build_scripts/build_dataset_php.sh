#!/bin/bash
# Script to collect all merges on a PHP dataset

dataset_build_scripts/build_dataset.sh input_data/repos_reaper_PHP_0_1000.csv merges/repos_reaper_php -g -m -b --max_num_merges 25 --language php --test_size 1.0
