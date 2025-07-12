#!/bin/bash
# Main script to build all datasets
# This script runs all individual dataset build scripts

echo "Starting to build all datasets..."
echo "================================"

# C dataset
echo "Building C dataset..."
./dataset_build_scripts/build_dataset_c.sh

# C++ dataset
echo "Building C++ dataset..."
./dataset_build_scripts/build_dataset_cpp.sh

# C# dataset
echo "Building C# dataset..."
./dataset_build_scripts/build_dataset_csharp.sh

# Python dataset
echo "Building Python dataset..."
./dataset_build_scripts/build_dataset_python.sh

# Ruby dataset
echo "Building Ruby dataset..."
./dataset_build_scripts/build_dataset_ruby.sh

# PHP dataset
echo "Building PHP dataset..."
./dataset_build_scripts/build_dataset_php.sh

# JavaScript datasets (both use GitHub data)
echo "Building JavaScript dataset (GitHub 1000 repos)..."
./dataset_build_scripts/build_dataset_javascript.sh

# Note: The github_javascript script uses the same data
# echo "Building JavaScript dataset (GitHub 1000 repos)..."
# ./dataset_build_scripts/build_dataset_github_javascript.sh

# TypeScript dataset
echo "Building TypeScript dataset (GitHub 1000 repos)..."
./dataset_build_scripts/build_dataset_typescript.sh

# Go datasets (both use GitHub data)
echo "Building Go dataset (GitHub 1000 repos)..."
./dataset_build_scripts/build_dataset_go.sh

# Note: The github_go script uses the same data
# echo "Building Go dataset (GitHub 1000 repos)..."
# ./dataset_build_scripts/build_dataset_github_go.sh

# Rust dataset
echo "Building Rust dataset (GitHub 1000 repos)..."
./dataset_build_scripts/build_dataset_rust.sh

# Java datasets (commented out as requested)
# echo "Building Reaper Java 1000 dataset..."
# ./dataset_build_scripts/build_dataset_reaper_java_1000.sh

# echo "Building Reaper Java 1000-1200 dataset..."
# ./dataset_build_scripts/build_dataset_reaper_java_1000_1200.sh

# Small test dataset
echo "Building small test dataset..."
./dataset_build_scripts/build_dataset_small.sh

echo "================================"
echo "All datasets build process completed!"
echo "Check individual logs for any errors."
