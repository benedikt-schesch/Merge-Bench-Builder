#!/bin/bash
# Script to collect all merges on a small dataset

dataset_build_scripts/build_dataset.sh input_data/repos_reaper_Python_0_1200.csv merges/repos_reaper_python "$@" --max_num_merges 1 --language python
