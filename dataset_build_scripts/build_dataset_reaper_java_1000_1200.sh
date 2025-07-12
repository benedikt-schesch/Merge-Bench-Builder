#!/bin/bash
# Script to build dataset for Java repositories 1000-1200 from Reaper
# This script runs all three steps: get conflicts, metrics, and build dataset

./dataset_build_scripts/build_dataset.sh -g -m -b input_data/repos_reaper_Java_1000_1200.csv output_data/reaper_java_1000_1200 --language java
