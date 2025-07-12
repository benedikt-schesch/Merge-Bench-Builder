#!/bin/bash

# Build dataset for top 1000 JavaScript repositories from GitHub
# This script runs all three steps: get conflicts, metrics, and build dataset
./dataset_build_scripts/build_dataset.sh -g -m -b input_data/repos_github_Javascript_0_1000.csv output_data/github_javascript_1000 --language javascript
