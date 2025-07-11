#!/bin/bash
# Script to collect all merges on a C# dataset

dataset_build_scripts/build_dataset.sh "input_data/repos_reaper_C#_0_1200.csv" merges/repos_reaper_csharp "$@" --max_num_merges 100 --language csharp --test_size 1.0
