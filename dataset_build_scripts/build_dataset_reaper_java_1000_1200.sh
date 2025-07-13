#!/bin/bash
# Script to build dataset for Java repositories 1000-1200 from Reaper

./dataset_build_scripts/build_dataset.sh input_data/repos_reaper_Java_1000_1200.csv merges/repos_reaper_java_test -g -m -b --language java
