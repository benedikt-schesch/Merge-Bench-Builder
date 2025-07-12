#!/bin/bash
# Script to collect all merges on a small dataset

dataset_build_scripts/build_dataset.sh input_data/repos_reaper_Python_0_1000.csv merges/repos_reaper_python "$@" --max_num_merges 100 --language python --test_size 1.0
