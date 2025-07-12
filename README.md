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

- Python 3.12 or later
- Git
- Sufficient disk space for repository cloning and dataset storage

## Installation ⚙️

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/Merge-Bench-Builder.git
   cd Merge-Bench-Builder
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install --upgrade pip
   pip install -e .
   ```

## Usage

### Quick Start - Small Test Dataset

```bash
./dataset_build_scripts/build_dataset_small.sh -g -m -b
```

### Full Dataset Construction

```bash
# Build dataset with 1000 merges per repository
./dataset_build_scripts/build_dataset_reaper_1000.sh -g -m -b

# Build dataset with 100 merges per repository
./dataset_build_scripts/build_dataset_reaper_100.sh -g -m -b
```

### Language-Specific Dataset Construction

```bash
# Build C dataset
./dataset_build_scripts/build_dataset_c.sh -g -m -b

# Build C++ dataset
./dataset_build_scripts/build_dataset_cpp.sh -g -m -b

# Build C# dataset
./dataset_build_scripts/build_dataset_csharp.sh -g -m -b

# Build Go dataset (from GitHub)
./dataset_build_scripts/build_dataset_go.sh -g -m -b
# Or use the specific GitHub Go 1000 repos dataset
./dataset_build_scripts/build_dataset_github_go_1000.sh -g -m -b

# Build Java dataset (first 1000 repos)
./dataset_build_scripts/build_dataset_reaper_java_1000.sh -g -m -b
# Build Java dataset (repos 1000-1200)
./dataset_build_scripts/build_dataset_reaper_java_1000_1200.sh -g -m -b

# Build JavaScript dataset (from GitHub)
./dataset_build_scripts/build_dataset_javascript.sh -g -m -b
# Or use the specific GitHub JavaScript 1000 repos dataset
./dataset_build_scripts/build_dataset_github_javascript_1000.sh -g -m -b

# Build PHP dataset
./dataset_build_scripts/build_dataset_php.sh -g -m -b

# Build Python dataset
./dataset_build_scripts/build_dataset_python.sh -g -m -b

# Build Ruby dataset
./dataset_build_scripts/build_dataset_ruby.sh -g -m -b

# Build Rust dataset (from GitHub)
./dataset_build_scripts/build_dataset_rust.sh -g -m -b

# Build TypeScript dataset (from GitHub)
./dataset_build_scripts/build_dataset_typescript.sh -g -m -b
```

### Custom Repository Sampling

#### From Reaper Dataset

```bash
# Sample C repositories
python src/sample_reaper_repos.py --language C --n 1000 --start_index 0

# Sample C++ repositories
python src/sample_reaper_repos.py --language "C++" --n 1000 --start_index 0

# Sample C# repositories
python src/sample_reaper_repos.py --language "C#" --n 1000 --start_index 0

# Sample Java repositories
python src/sample_reaper_repos.py --language Java --n 1000 --start_index 0

# Sample PHP repositories
python src/sample_reaper_repos.py --language PHP --n 1000 --start_index 0

# Sample Python repositories
python src/sample_reaper_repos.py --language Python --n 1000 --start_index 0

# Sample Ruby repositories
python src/sample_reaper_repos.py --language Ruby --n 1000 --start_index 0

# Sample from custom input file
python src/sample_reaper_repos.py --input_path input_data/custom_repos.csv --language JavaScript --n 200
```

#### From GitHub API

```bash
# Download JavaScript repositories from GitHub (requires GitHub token for better rate limits)
python src/download_github_repos.py --language javascript --max-results 1200 --exclude-archived

# With GitHub token (recommended)
python src/download_github_repos.py --token YOUR_GITHUB_TOKEN --language javascript --max-results 1200 --exclude-archived

# Or set GITHUB_TOKEN environment variable
export GITHUB_TOKEN=your_token_here
python src/download_github_repos.py --language javascript --max-results 1200 --exclude-archived

# Download other languages from GitHub
python src/download_github_repos.py --language typescript --max-results 1200 --exclude-archived
python src/download_github_repos.py --language go --max-results 1200 --exclude-archived
python src/download_github_repos.py --language rust --max-results 1200 --exclude-archived

# With custom parameters
python src/download_github_repos.py --language go --max-results 1000 --min-stars 500
python src/download_github_repos.py --language rust --max-results 800 --min-stars 1000
```

### Script Options

All build scripts support these flags:
- `-g`: Run repository download and conflict extraction steps
- `-m`: Compute dataset metrics and statistics
- `-b`: Build the final processed dataset
- `--test_size <fraction>`: Fraction reserved for testing (default: 0.2)
- `--max_num_merges <n>`: Maximum merges to collect per repository (default: 100)

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

### 4. Quality Control
- Validate extracted conflicts
- Remove duplicates and low-quality samples
- Ensure proper formatting and structure

## Project Structure

```
.
├── dataset_build_scripts/          # Dataset building scripts
│   ├── build_dataset.sh           # Main dataset building script
│   ├── build_dataset_small.sh     # Small test dataset
│   ├── build_dataset_c.sh         # C dataset builder
│   ├── build_dataset_cpp.sh       # C++ dataset builder
│   ├── build_dataset_csharp.sh    # C# dataset builder
│   ├── build_dataset_go.sh        # Go dataset builder
│   ├── build_dataset_javascript.sh # JavaScript dataset builder
│   ├── build_dataset_php.sh       # PHP dataset builder
│   ├── build_dataset_python.sh    # Python dataset builder
│   ├── build_dataset_ruby.sh      # Ruby dataset builder
│   ├── build_dataset_rust.sh      # Rust dataset builder
│   ├── build_dataset_typescript.sh # TypeScript dataset builder
│   ├── build_dataset_reaper_100.sh # 100 merges per repo
│   ├── build_dataset_reaper_1000.sh # 1000 merges per repo
│   ├── build_dataset_reaper_java_1000.sh # Java first 1000 repos
│   └── build_dataset_reaper_java_1000_1200.sh # Java repos 1000-1200
├── src/                           # Core source code
│   ├── build_dataset.py          # Dataset building logic
│   ├── download_github_repos.py  # GitHub API repository fetcher
│   ├── download_reaper_dataset.py # Reaper dataset downloader
│   ├── get_conflict_files.py     # Conflict extraction
│   ├── find_merges.py            # Merge commit discovery
│   ├── extract_conflict_blocks.py # Conflict block processing
│   ├── metrics_conflict_blocks.py # Dataset metrics
│   ├── sample_reaper_repos.py    # Repository sampling
│   └── utils.py                  # Utility functions
├── input_data/                   # Input datasets and repository lists
├── setup/                        # Setup and configuration scripts
├── download_reaper_dataset.py    # Reaper dataset downloader
├── pyproject.toml               # Project configuration
├── README.md                    # This file
└── LICENSE                      # MIT License
```

## Configuration

### Repository Filtering
- **Language**: Filter by programming language (C, C++, C#, Java, PHP, Python, Ruby, JavaScript, etc.)
- **Stars**: Minimum star count threshold
- **Quality metrics**: Architecture, documentation, test coverage scores

### Dataset Parameters
- **Size**: Number of repositories to process
- **Merge limit**: Maximum merges per repository
- **Test split**: Fraction of data reserved for testing

### Output Formats
- CSV files with conflict metadata
- JSON files with detailed conflict information
- Processed datasets ready for analysis

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
