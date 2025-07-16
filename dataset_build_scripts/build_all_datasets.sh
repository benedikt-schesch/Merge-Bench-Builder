#!/bin/bash
# Main script to build all datasets
# This script runs all individual dataset build scripts

# Accept command line arguments, default to "-g -m -b" if none provided
if [ $# -eq 0 ]; then
    args="-g -m -b"
else
    args="$@"
fi

echo "Starting to build all datasets with arguments: $args"
echo "================================"

# Arrays to store results for final summary
declare -a LANGUAGES=()
declare -a TEST_SIZES=()
declare -a REPO_COUNTS=()
declare -a BUILD_STATUS=()

# Function to run a dataset build script and extract the test set size and repository count
run_dataset_build() {
    local script_name=$1
    local language_name=$2

    echo "Building $language_name dataset..."

    # Capture the output and run the script with arguments
    output=$(./dataset_build_scripts/$script_name $args 2>&1)
    exit_code=$?

    # Display the output
    echo "$output"

    # Extract test set size
    test_size=$(echo "$output" | grep -o "Test set size: [0-9]*" | grep -o "[0-9]*")

    # Extract repository count
    repo_count=$(echo "$output" | grep -o "Repositories contributing to final dataset: [0-9]*" | grep -o "[0-9]*")

    # Store results for summary
    LANGUAGES+=("$language_name")
    TEST_SIZES+=("${test_size:-N/A}")
    REPO_COUNTS+=("${repo_count:-N/A}")
    if [ $exit_code -eq 0 ]; then
        BUILD_STATUS+=("✓")
    else
        BUILD_STATUS+=("✗")
    fi

    # Display summary with both metrics
    if [ -n "$test_size" ] && [ -n "$repo_count" ]; then
        echo "→ $language_name dataset completed with test set size: $test_size, unique repos: $repo_count"
    elif [ -n "$test_size" ]; then
        echo "→ $language_name dataset completed with test set size: $test_size (repo count not found)"
    elif [ -n "$repo_count" ]; then
        echo "→ $language_name dataset completed with unique repos: $repo_count (test set size not found)"
    else
        echo "→ $language_name dataset completed (metrics not found)"
    fi
    echo ""

    return $exit_code
}

# C dataset
run_dataset_build "build_dataset_c.sh" "C"

# C++ dataset
run_dataset_build "build_dataset_cpp.sh" "C++"

# C# dataset
run_dataset_build "build_dataset_csharp.sh" "C#"

# Python dataset
run_dataset_build "build_dataset_python.sh" "Python"

# Ruby dataset
run_dataset_build "build_dataset_ruby.sh" "Ruby"

# PHP dataset
run_dataset_build "build_dataset_php.sh" "PHP"

# JavaScript datasets (both use GitHub data)
run_dataset_build "build_dataset_javascript.sh" "JavaScript"

# TypeScript dataset
run_dataset_build "build_dataset_typescript.sh" "TypeScript"

# Go datasets (both use GitHub data)
run_dataset_build "build_dataset_go.sh" "Go"

# Rust dataset
run_dataset_build "build_dataset_rust.sh" "Rust"

# Java datasets (We ran those before the others)
# echo "Building Reaper Java 1000 dataset..."
# ./dataset_build_scripts/build_dataset_reaper_java_1000.sh

# echo "Building Reaper Java 1000-1200 dataset..."
# ./dataset_build_scripts/build_dataset_reaper_java_1000_1200.sh

echo "================================"
echo "FINAL SUMMARY"
echo "================================"

# Display summary table
printf "%-12s %-8s %-12s %-12s\n" "Language" "Status" "Test Size" "Unique Repos"
printf "%-12s %-8s %-12s %-12s\n" "--------" "------" "---------" "------------"

# Calculate totals
total_test_size=0
total_unique_repos=0
successful_builds=0

for i in "${!LANGUAGES[@]}"; do
    language="${LANGUAGES[i]}"
    status="${BUILD_STATUS[i]}"
    test_size="${TEST_SIZES[i]}"
    repo_count="${REPO_COUNTS[i]}"

    printf "%-12s %-8s %-12s %-12s\n" "$language" "$status" "$test_size" "$repo_count"

    # Add to totals if values are numbers
    if [[ "$test_size" =~ ^[0-9]+$ ]]; then
        total_test_size=$((total_test_size + test_size))
    fi
    if [[ "$repo_count" =~ ^[0-9]+$ ]]; then
        total_unique_repos=$((total_unique_repos + repo_count))
    fi
    if [[ "$status" == "✓" ]]; then
        successful_builds=$((successful_builds + 1))
    fi
done

echo ""
echo "TOTALS:"
echo "- Successful builds: $successful_builds/${#LANGUAGES[@]}"
echo "- Total test set size: $total_test_size"
echo "- Total unique repositories: $total_unique_repos"

echo ""
echo "All datasets build process completed!"
echo "Check individual logs for any errors."
