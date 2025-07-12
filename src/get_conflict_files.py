#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=too-many-arguments, too-many-locals, too-many-positional-arguments
"""
Usage:
    python3 extract_conflict_files.py \
        --repos path/to/repos.csv \
        --output_dir path/to/conflict_files \
        --n_threads 8

Modified to use a two-step process:
1. Collect all merges in parallel (one task per repository)
2. Process conflict files in parallel (one task per merge)

Output files are named as:
    merge_id + letter .conflict
    merge_id + letter .final_merged
and a new CSV is dumped mapping each merge (using its merge_id) to the comma-separated
list of conflict file IDs.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
import os
import shutil
from pathlib import Path
from typing import List, Optional
import pandas as pd
from git import GitCommandError, Repo
from loguru import logger
from tqdm import tqdm
import timeout_decorator
from find_merges import get_repo, get_merges


def get_num_workers(n_threads: Optional[int] = None) -> int:
    """Get the number of workers for parallel processing."""
    if n_threads is not None and n_threads > 0:
        return n_threads
    os_cpu_count = os.cpu_count()
    if os_cpu_count is None:
        return 1
    return os_cpu_count - 1


logger.add("run.log", backtrace=True, diagnose=True)

WORKING_DIR = Path(".workdir")


def get_file_extensions(language: str) -> list[str]:
    """Get the file extensions for a given programming language."""
    language_extensions = {
        "java": [".java"],
        "python": [".py"],
        "javascript": [".js"],
        "typescript": [".ts", ".tsx"],
        "cpp": [".cpp", ".cc", ".cxx", ".h", ".hpp"],
        "csharp": [".cs"],
        "php": [".php"],
        "ruby": [".rb"],
        "c": [".c", ".h"],
        "go": [".go"],
        "rust": [".rs"],
    }
    return language_extensions.get(language.lower(), [".java"])


def concatenate_csvs(input_path: Path) -> pd.DataFrame:
    """
    Finds all CSV files in the given directory and its subdirectories, concatenates them,
    and saves the merged file to the specified output path.

    :param input_path: Directory containing CSV files.
    """
    csv_files = list(Path(input_path).rglob("*.csv"))

    if not csv_files:
        raise ValueError(f"No CSV files found in {input_path}")

    dataframes = [pd.read_csv(file) for file in csv_files]
    merged_df = pd.concat(dataframes, ignore_index=True)
    return merged_df


def checkout_commit(repo: Repo, commit_sha: str, branch_name: str) -> None:
    """Checkout a commit by SHA and create a new branch."""
    git = repo.git
    try:
        git.branch("-D", branch_name, force=True)
    except GitCommandError:
        pass
    git.checkout(commit_sha, b=branch_name)


def copy_conflicting_files_and_goal(
    conflict_files: List[Path],
    final_repo: Repo,
    final_commit_sha: str,
    output_dir: Path,
    merge_id: int,
) -> List[str]:
    """
    Copy conflict-marked files from the repo working tree, and also
    copy the final merged ("goal") file from the real merged commit.
    Files are written into output_dir with names: merge_id-index.conflict
    and merge_id-index.final_merged.
    Returns the list of conflict IDs (e.g. ["1-0", "1-1", ...]) for this merge.
    """
    conflicts: List[str] = []
    for conflict_num, conflict_file in enumerate(conflict_files):
        conflict_id = f"{merge_id}-{conflict_num}"

        # 1) conflict-marked file from local working tree
        conflict_file_path = Path(final_repo.working_dir) / conflict_file
        if not conflict_file_path.is_file():
            logger.warning(
                f"Conflict file not found in working tree: {conflict_file_path}"
            )
            continue
        conflict_marked_target = output_dir / f"{conflict_id}.conflict"
        conflict_marked_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(conflict_file_path, conflict_marked_target)
        conflicts.append(conflict_id)

        # 2) final merged version from the real merged commit
        final_file_target = output_dir / f"{conflict_id}.final_merged"
        try:
            content = final_repo.git.show(f"{final_commit_sha}:{conflict_file}")
            final_file_target.parent.mkdir(parents=True, exist_ok=True)
            final_file_target.write_text(content, encoding="utf-8")
        except GitCommandError:
            logger.warning(
                f"File {conflict_file_path} not found in final merged commit {final_commit_sha}"
            )
    return conflicts


@timeout_decorator.timeout(5 * 60, use_signals=False)
def reproduce_merge_and_extract_conflicts(
    repo_slug: str,
    left_sha: str,
    right_sha: str,
    merge_sha: str,
    output_dir: Path,
    merge_id: int,
    conflict_cache_folder: Path,
    resolved_merge_cache_folder: Path,
    file_extensions: List[str] = [".java"],
) -> List[str]:
    """
    Checkout left_sha, merge right_sha.
    If conflicts occur, copy conflict-marked files and final merged files to output_dir.
    Additionally, cache these files under "merge_cache/conflicts" using the original
    filenames. When copying them to output_dir, the files are renamed using merge_id-index.
    If the cache already exists, the cached files are simply copied over.
    Returns the list of conflict IDs for this merge.
    """
    # Check if cache exists (both subdirectories must be present)
    if conflict_cache_folder.exists() and resolved_merge_cache_folder.exists():
        cached_conflict_files = sorted(conflict_cache_folder.iterdir())
        conflicts = []
        for i, cached_file in enumerate(cached_conflict_files):
            conflict_id = f"{merge_id}-{i}"
            destination_conflict = output_dir / f"{conflict_id}.conflict"
            destination_final = output_dir / f"{conflict_id}.final_merged"
            destination_conflict.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cached_file, destination_conflict)
            # Look for corresponding final merged file using the same original filename
            final_file = resolved_merge_cache_folder / cached_file.name
            if final_file.is_file():
                shutil.copy2(final_file, destination_final)
            conflicts.append(conflict_id)
        return conflicts

    # No cache exists, so reproduce the merge.
    logger.info(f"Reproducing merge {merge_sha} for {repo_slug}")
    conflict_cache_folder.mkdir(parents=True, exist_ok=True)
    resolved_merge_cache_folder.mkdir(parents=True, exist_ok=True)
    repo = get_repo(repo_slug)
    temp_dir = WORKING_DIR / f"{repo_slug}_merge_{merge_id}"
    shutil.copytree(repo.working_dir, temp_dir, dirs_exist_ok=True)
    repo = Repo(temp_dir)
    repo.git.checkout(left_sha, force=True)
    conflict_files: List[Path] = []
    try:
        repo.git.merge(right_sha)
    except GitCommandError:
        status_output = repo.git.status("--porcelain")
        for line in status_output.splitlines():
            if line.startswith("UU "):
                path_part = line[3:].strip()
                # Check if file matches any of the target extensions
                if any(path_part.endswith(ext) for ext in file_extensions):
                    conflict_files.append(Path(path_part))

    result: List[str] = []
    if conflict_files:
        logger.info(
            f"Conflict in {left_sha} + {right_sha} => {merge_sha}, files: {conflict_files}"
        )
        # Sort conflict files by name deterministically
        conflict_files = sorted(conflict_files)
        result = copy_conflicting_files_and_goal(
            conflict_files=conflict_files,
            final_repo=repo,
            final_commit_sha=merge_sha,
            output_dir=output_dir,
            merge_id=merge_id,
        )
        for i, conflict_file in enumerate(conflict_files):
            conflict_id = f"{merge_id}-{i}"
            original_name = conflict_file.name
            # Cache conflict-marked file
            src_conflict = output_dir / f"{conflict_id}.conflict"
            dst_conflict = conflict_cache_folder / original_name
            if src_conflict.is_file():
                shutil.copy2(src_conflict, dst_conflict)
            # Cache final merged file
            src_final = output_dir / f"{conflict_id}.final_merged"
            dst_final = resolved_merge_cache_folder / original_name
            if src_final.is_file():
                shutil.copy2(src_final, dst_final)
    return result


def collect_merges(
    repo_slug: str, output_dir: Path, max_num_merges: int, max_branches: int = 1000
) -> pd.DataFrame:
    """
    Step 1: Collect merges for a single repository.
    """
    try:
        repo = get_repo(repo_slug)
    except Exception as e:
        logger.error(f"Error getting repo {repo_slug}: {e}")
        return pd.DataFrame()
    merges = get_merges(repo, repo_slug, output_dir / "merges", max_num_merges, max_branches)
    return merges


def process_merge(merge_row, output_dir: Path, file_extensions: List[str] = [".java"]) -> tuple:
    """
    Step 2: Process a single merge to extract conflict files.
    """
    merge_id = merge_row.name
    repo_slug = merge_row["repository"]
    cache_folder = Path("merge_cache/conflicts") / repo_slug / merge_row["merge_commit"]
    conflict_cache_folder = cache_folder / "conflict"
    resolved_merge_cache_folder = cache_folder / "final_merged"
    try:
        conflicts = reproduce_merge_and_extract_conflicts(
            repo_slug=merge_row["repository"],
            left_sha=merge_row["parent_1"],
            right_sha=merge_row["parent_2"],
            merge_sha=merge_row["merge_commit"],
            output_dir=output_dir / "conflict_files",
            merge_id=merge_id,
            conflict_cache_folder=conflict_cache_folder,
            resolved_merge_cache_folder=resolved_merge_cache_folder,
            file_extensions=file_extensions,
        )
        conflict_str = ";".join(conflicts)
        shutil.rmtree(WORKING_DIR / f"{repo_slug}_merge_{merge_id}", ignore_errors=True)
        return merge_id, conflict_str
    except timeout_decorator.TimeoutError:
        logger.error(f"Timeout for merge {merge_row['merge_commit']} in {repo_slug}")
        # The content of the cache folder might be corrupt, so delete the content of it
        # but keep the folder itself to avoid re-creating it
        if conflict_cache_folder.exists():
            for file in conflict_cache_folder.iterdir():
                file.unlink()
        if resolved_merge_cache_folder.exists():
            for file in resolved_merge_cache_folder.iterdir():
                file.unlink()
    except Exception as e:
        logger.error(
            f"Error processing merge {merge_row['merge_commit']} in {repo_slug}: {e}"
        )
    shutil.rmtree(WORKING_DIR / f"{repo_slug}_merge_{merge_id}", ignore_errors=True)
    return merge_id, ""


def clone_single_repository(repo_slug: str) -> str:
    """
    Clone a single repository. Used for parallel processing.
    Returns the repo_slug for tracking purposes.
    """
    try:
        get_repo(repo_slug, log=True)
        return repo_slug
    except Exception as e:
        logger.error(f"Failed to clone {repo_slug}: {e}")
        return f"FAILED: {repo_slug}"


def clone_all_repositories(repos_df: pd.DataFrame, num_workers: int) -> None:
    """
    Step 0: Clone all repositories in parallel with progress bar.
    This ensures all repositories are available locally before processing begins.
    """
    logger.info(f"Step 0: Cloning {len(repos_df)} repositories using {num_workers} threads...")
    
    repo_slugs = repos_df["repository"].tolist()
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all cloning tasks
        futures = {executor.submit(clone_single_repository, repo_slug): repo_slug 
                  for repo_slug in repo_slugs}
        
        # Process results with progress bar
        for future in tqdm(as_completed(futures), total=len(repo_slugs), desc="Cloning repos"):
            result = future.result()
            if result.startswith("FAILED:"):
                logger.warning(f"Cloning failed for {result[7:]}")


def main():
    """Main function with three steps: clone repositories, collect merges, then extract conflict files."""
    parser = argparse.ArgumentParser(description="Extract conflict files from merges.")
    parser.add_argument(
        "--repos",
        default="input_data/repos_small.csv",
        help="CSV with merges (org/repo, merge_commit, parent_1, parent_2)",
    )
    parser.add_argument(
        "--output_dir",
        default="merges/repos_small",
        help="Directory to store conflict files",
    )
    parser.add_argument(
        "--n_threads",
        type=int,
        default=None,
        help="Number of parallel threads (if not specified: use all CPU cores)",
    )
    parser.add_argument(
        "--seed",
        required=False,
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--max_num_merges",
        type=int,
        default=100,
        help="Maximum number of merges to process per repository (default: 100)",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="java",
        choices=["java", "python", "javascript", "typescript", "cpp", "csharp", "php", "ruby", "c", "go", "rust"],
        help="Programming language to filter conflict files (default: java)",
    )
    parser.add_argument(
        "--max_branches",
        type=int,
        default=1000,
        help="Maximum number of branches to process per repository (default: 1000)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "conflict_files/cache").mkdir(parents=True, exist_ok=True)

    repos_df = pd.read_csv(args.repos)
    num_workers = get_num_workers(args.n_threads)

    # Ensure deterministic order of repositories
    repos_df = repos_df.sort_values(by="repository")

    # STEP 0: Clone all repositories first
    clone_all_repositories(repos_df, num_workers)

    # STEP 1: Collect all merges in parallel
    logger.info(
        f"Step 1: Collecting merges for {len(repos_df)} repos using {num_workers} threads..."
    )

    # For deterministic results, process in ordered batches
    result = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        tasks = {
            executor.submit(
                collect_merges, row["repository"], output_dir, args.max_num_merges, args.max_branches
            ): i
            for i, (_, row) in enumerate(repos_df.iterrows())
        }

        # Process results in deterministic order based on input order
        futures_by_index = sorted([(index, future) for future, index in tasks.items()])
        ordered_futures = [f for _, f in futures_by_index]

        for future in tqdm(as_completed(tasks), total=len(repos_df), desc="Collecting merges"):
            result.append(future.result())

    # Combine all merge CSVs
    all_merges_df = pd.concat(result)
    all_merges_df["merge_idx"] = range(0, len(all_merges_df))
    all_merges_df.set_index("merge_idx", inplace=True)
    all_merges_df.to_csv(output_dir / "all_merges.csv")
    logger.info(f"Found {len(all_merges_df)} merges in total.")
    # Make sure merge_commit is unique
    if all_merges_df[["repository", "merge_commit"]].duplicated().any():
        logger.error("Duplicate merge_commit found in all_merges.csv")
        sys.exit(1)

    # Sort the merge dataframe to ensure deterministic order
    all_merges_df = all_merges_df.sort_values(by=["repository", "merge_commit"])
    all_merges_df["merge_idx"] = range(0, len(all_merges_df))
    all_merges_df.set_index("merge_idx", inplace=True)

    # Get file extensions based on language
    file_extensions = get_file_extensions(args.language)
    logger.info(f"Processing conflicts for {args.language} files ({file_extensions})")

    # STEP 2: Process each merge in parallel to extract conflict files
    logger.info(
        f"Step 2: Processing {len(all_merges_df)} merges for "
        f"conflicts using {num_workers} threads..."
    )
    all_merges_df["conflicts"] = ""

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures_dict = {
            executor.submit(process_merge, merge_row, output_dir, file_extensions): merge_id
            for merge_id, merge_row in all_merges_df.iterrows()
        }

        # Process results in deterministic order
        ordered_items = sorted(
            [(merge_id, future) for future, merge_id in futures_dict.items()]
        )
        ordered_futures = [(future, merge_id) for merge_id, future in ordered_items]

        # Process each future in deterministic order
        for future, merge_id in tqdm(ordered_futures):
            _, conflict_str = future.result()
            all_merges_df.loc[merge_id, "conflicts"] = conflict_str  # type: ignore

    # Combine all results
    all_merges_df = all_merges_df[all_merges_df["conflicts"] != ""]  # pylint: disable=use-implicit-booleaness-not-comparison-to-string
    all_merges_df.to_csv(output_dir / "conflict_files.csv")

    logger.info("Done extracting conflict files.")


if __name__ == "__main__":
    main()
