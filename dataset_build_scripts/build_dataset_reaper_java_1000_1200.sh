#!/bin/bash
# Script to build dataset for Java repositories 1000-1200 from Reaper

python3 src/build_dataset.py \
    --repo-csv-path input_data/repos_reaper_Java_1000_1200.csv \
    --output-dir output_data/reaper_java_1000_1200 \
    --num-processes 8
