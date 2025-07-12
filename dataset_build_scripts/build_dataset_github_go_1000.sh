#!/bin/bash

# Build dataset for top 1000 Go repositories from GitHub
python3 src/build_dataset.py \
    --repo-csv-path input_data/repos_github_Go_0_1000.csv \
    --output-dir output_data/github_go_1000 \
    --num-processes 8
