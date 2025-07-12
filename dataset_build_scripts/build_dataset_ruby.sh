#!/bin/bash
# Script to collect all merges on a Ruby dataset

dataset_build_scripts/build_dataset.sh input_data/repos_reaper_Ruby_0_1000.csv merges/repos_reaper_ruby "$@" --max_num_merges 100 --language ruby --test_size 1.0
