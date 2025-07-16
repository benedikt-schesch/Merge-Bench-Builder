#!/bin/bash
# Script to collect all merges on a small dataset

# Pass through all command line arguments to build_dataset.sh
dataset_build_scripts/build_dataset.sh input_data/repos_small.csv merges/repos_small "$@" -keep_trivial_resolution
