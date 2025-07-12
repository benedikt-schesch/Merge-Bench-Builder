#!/bin/bash

# Build dataset for top 1000 JavaScript repositories from GitHub
python3 src/build_dataset.py \
    --repo-csv-path input_data/repos_github_Javascript_0_1000.csv \
    --output-dir output_data/github_javascript_1000 \
    --num-processes 8
