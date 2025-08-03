# Merge-Bench-Builder

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/benedikt-schesch/Merge-Bench-Builder/actions/workflows/ci.yml/badge.svg)](https://github.com/benedikt-schesch/Merge-Bench-Builder/actions/workflows/ci.yml)

A toolkit for constructing merge conflict datasets from Git repositories. This tool helps researchers and developers build comprehensive datasets for studying merge conflict resolution patterns. 🛠️

## Table of Contents

- [Features ✨](#features)
- [Prerequisites 📋](#prerequisites)
- [Installation ⚙️](#installation)
- [Usage](#usage)
- [Dataset Construction 🗂️](#dataset-construction)
- [Project Structure](#project-structure)
- [License](#license)

## Features ✨

- 🛠️ Build customizable merge conflict datasets from Git history
- 📊 Extract and analyze merge conflicts from real repositories
- 🔍 Filter repositories by programming language, stars, and other criteria
- 📈 Compute conflict metrics and analyze resolution patterns
- ⚙️ Support for various dataset sizes with configurable parameters
- 🌐 Download and process repositories from the Reaper dataset

## Prerequisites 📋

- [uv](https://docs.astral.sh/uv/) - Python package manager
- Sufficient disk space for repository cloning and dataset storage (2TB+)

## Installation ⚙️

1. Clone the repository:

   ```bash
   git clone https://github.com/benedikt-schesch/Merge-Bench-Builder.git
   cd Merge-Bench-Builder
   ```

2. Install dependencies:

   ```bash
   uv sync
   ```

## Usage

### Build All Datasets

The main script to build datasets for all supported programming languages:

```bash
./dataset_build_scripts/build_all_datasets.sh
```

This script builds datasets for the following languages:
- C, C++, C#
- Python, Ruby, PHP
- JavaScript, TypeScript
- Go, Rust

### Advanced Usage

#### Custom Repository Sampling

```bash
# Sample specific language repositories from Reaper dataset
python src/sample_reaper_repos.py --language Java --n 1000 --start_index 0

# Download repositories from GitHub API
export GITHUB_TOKEN=your_token_here
python src/download_github_repos.py --language javascript --max-results 1000 --exclude-archived
```

### Script Options

```bash
./dataset_build_scripts/build_all_datasets.sh
```

Supports the following flags:

All build scripts support these flags:
- `-g`: Run repository download and conflict extraction steps
- `-m`: Compute dataset metrics and statistics
- `-b`: Build the final processed dataset

## Dataset Construction 🗂️

The dataset construction process involves several stages:

### 1. Repository Selection
- Download repository metadata from the Reaper dataset
- Filter repositories by language, stars, and quality metrics
- Sample repositories based on specified criteria

### 2. Merge Conflict Extraction
- Clone selected repositories
- Analyze Git history to find merge commits
- Extract merge conflicts and their resolutions
- Process conflict blocks and surrounding context

### 3. Dataset Processing
- Clean and normalize conflict data
- Compute metrics and statistics
- Split data into training/testing sets
- Generate final dataset files

### 4. Filtering
- Validate extracted conflicts
- Remove duplicates and low-quality samples
- Ensure proper formatting and structure

## Project Structure

```
.
├── dataset_build_scripts/          # Dataset building scripts
│   ├── build_all_datasets.sh      # Main script - builds all language datasets
│   └── build_dataset_*.sh         # Individual language-specific scripts
├── src/                           # Core source code
│   ├── build_dataset.py          # Dataset building logic
│   ├── download_github_repos.py  # GitHub API repository fetcher
│   ├── sample_reaper_repos.py    # Repository sampling from Reaper dataset
│   ├── get_conflict_files.py     # Conflict extraction
│   ├── find_merges.py            # Merge commit discovery
│   ├── extract_conflict_blocks.py # Conflict block processing
│   ├── metrics_conflict_blocks.py # Dataset metrics
│   └── utils.py                  # Utility functions
├── input_data/                   # Input datasets and repository lists
├── merges/                       # Generated datasets (created after running scripts)
├── pyproject.toml               # Project configuration
├── README.md                    # This file
└── LICENSE                      # MIT License
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
