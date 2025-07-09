#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CreateConflictMetrics.py

Usage:
    python CreateConflictMetrics.py \
        --input_dir /path/to/output_folder \
        --csv_out /path/to/output_metrics.csv \
        --max_line_count 20 \
        --keep_trivial_resolution \
        --selected_out_dir /path/to/selected_conflicts

What does this script do?
  - Finds *.conflict and *.resolved_conflict files in `--input_dir` (pairs).
  - For each pair <basename><n>.conflict / <basename><n>.resolved_conflict:
      1) Reads the conflict snippet (which includes context + conflict markers).
      2) Identifies:
           - context_before (lines above <<<<<<<),
           - conflict_block (lines from <<<<<<< up through >>>>>>>),
           - context_after (lines after >>>>>>>).
      3) Further splits the conflict block into:
           - left_parent (lines after <<<<<<< until a base marker or =======),
           - base (lines after a base marker “|||||||” until =======, if present),
           - right_parent (lines after ======= until >>>>>>>).
      4) Reads the resolved snippet (the entire file).
      5) Computes metrics (in lines) for each portion, plus token count of the full conflict.
      6) (New) Filters out conflicts based on conditions like trivial resolution, max lines, etc.
      7) (New) Copies selected conflict/resolution files to `--selected_out_dir`, if specified.
      8) Writes all metrics (with `selected` column) to a CSV file using pandas.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple
import shutil
import pandas as pd
from transformers import AutoTokenizer
from loguru import logger
from tqdm import tqdm
from build_dataset import build_query
from variables import MODEL_NAME, MAX_PROMPT_LENGTH


logger.add("run.log", backtrace=True, diagnose=True)


def get_token_count(tokenizer, conflict_query: List[str]) -> int:
    """Returns the token count of the conflict query (entire .conflict file)."""
    query = build_query("\n".join(conflict_query))
    return len(tokenizer(query)["input_ids"])


def split_single_conflict_snippet(
    lines: List[str],
) -> Tuple[List[str], List[str], List[str]]:
    """
    Given a list of lines that contain exactly one conflict block
    (which should have <<<<<<< ... >>>>>>>),
    returns (before_context, conflict_block, after_context).

    If markers are not found, raises a ValueError.
    """
    start_idx = -1
    end_idx = -1

    # Locate <<<<<<< marker
    for i, line in enumerate(lines):
        if line.startswith("<<<<<<<"):
            start_idx = i
            break

    if start_idx == -1:
        raise ValueError("Could not find start marker (<<<<<<<).")

    # Locate >>>>>>> marker
    for j in range(start_idx, len(lines)):
        if lines[j].startswith(">>>>>>>"):
            end_idx = j
            break

    if end_idx == -1:
        raise ValueError("Found start marker but no end marker (>>>>>>>).")

    before_context = lines[:start_idx]
    conflict_block = lines[start_idx : end_idx + 1]  # include end marker line
    after_context = lines[end_idx + 1 :]

    return before_context, conflict_block, after_context


def split_conflict_block(
    conflict_block: List[str],
) -> Tuple[List[str], List[str], Optional[List[str]]]:
    """
    Splits the conflict block into left_parent, right_parent, and base.

    The expected format is:
      <<<<<<< [optional identifier]
      (left parent lines)
      [optional base marker: "|||||||", followed by base lines]
      =======
      (right parent lines)
      >>>>>>>>

    If a base marker is not found, base is returned as None.
    """
    left_parent: List[str] = []
    base: List[str] = []
    right_parent: List[str] = []
    encountered_base_marker = False

    state = "left"
    for line in conflict_block:
        if line.startswith("<<<<<<<"):
            # skip the start marker line
            continue
        if line.startswith("|||||||"):
            state = "base"
            encountered_base_marker = True
            continue
        if line.startswith("======="):
            state = "right"
            continue
        if line.startswith(">>>>>>>"):
            break
        if state == "left":
            left_parent.append(line)
        elif state == "base":
            base.append(line)
        elif state == "right":
            right_parent.append(line)

    return left_parent, right_parent, None if not encountered_base_marker else base


def functional_equality(left: List[str], right: List[str]) -> bool:
    """
    Check if two lists of strings are functionally equivalent.
    """

    def normalize(lst: List[str]) -> str:
        return "".join(lst).replace(" ", "").replace("\t", "")

    return normalize(left) == normalize(right)


def load_conflict_files_mapping(conflict_files_csv_path: Path) -> pd.DataFrame:
    """Load the conflict_files.csv to map merge IDs to repositories."""
    try:
        df = pd.read_csv(conflict_files_csv_path)
        return df
    except Exception as e:
        logger.warning(f"Could not load conflict_files.csv from {conflict_files_csv_path}: {e}")
        return pd.DataFrame()


def extract_merge_id_from_conflict_id(conflict_id: str) -> str:
    """Extract merge ID from conflict ID (e.g., '123-0' -> '123')."""
    return conflict_id.split('-')[0]


def generate_repository_summary(df: pd.DataFrame, args, input_dir: Path) -> None:
    """Generate a repository-level summary CSV file."""
    
    # Auto-detect conflict_files.csv path if not provided
    if args.conflict_files_csv:
        conflict_files_csv_path = Path(args.conflict_files_csv)
    else:
        # Try to find it in the parent directories
        possible_paths = [
            input_dir.parent / "conflict_files.csv",
            input_dir.parent.parent / "conflict_files.csv",
        ]
        conflict_files_csv_path = None
        for path in possible_paths:
            if path.exists():
                conflict_files_csv_path = path
                break
    
    if not conflict_files_csv_path or not conflict_files_csv_path.exists():
        logger.warning("Could not find conflict_files.csv - repository summary will not include repository names")
        # Create a basic summary without repository mapping
        create_basic_repository_summary(df, args)
        return
    
    # Load the mapping from merge IDs to repositories
    conflict_files_df = load_conflict_files_mapping(conflict_files_csv_path)
    if conflict_files_df.empty:
        logger.warning("conflict_files.csv is empty - creating basic summary")
        create_basic_repository_summary(df, args)
        return
    
    # Extract merge IDs from conflict IDs and map to repositories
    df['merge_id'] = df['conflict_id'].apply(extract_merge_id_from_conflict_id)
    
    # Create mapping from merge_id to repository
    merge_to_repo = {}
    for _, row in conflict_files_df.iterrows():
        merge_idx = row.name if 'merge_idx' in conflict_files_df.columns or conflict_files_df.index.name == 'merge_idx' else row.get('merge_idx', row.name)
        repository = row['repository']
        merge_to_repo[str(merge_idx)] = repository
    
    # Map repositories to conflict data
    df['repository'] = df['merge_id'].map(merge_to_repo)
    
    # Group by repository and create summary
    repo_summary = []
    for repo, repo_df in df.groupby('repository'):
        if pd.isna(repo):
            repo = "UNKNOWN"
        
        total_conflicts = len(repo_df)
        selected_conflicts = repo_df['selected'].sum()
        failed_max_lines = repo_df['fail_max_lines'].sum()
        failed_token_count = repo_df['fail_token_count'].sum()
        failed_incoherent = repo_df['fail_incoherent'].sum()
        
        repo_summary.append({
            'repository': repo,
            'total_conflicts': total_conflicts,
            'selected_conflicts': selected_conflicts,
            'filtered_out_conflicts': total_conflicts - selected_conflicts,
            'failed_max_lines': failed_max_lines,
            'failed_token_count': failed_token_count,
            'failed_incoherent': failed_incoherent,
            'selection_rate': selected_conflicts / total_conflicts if total_conflicts > 0 else 0.0
        })
    
    # Create summary DataFrame and save
    summary_df = pd.DataFrame(repo_summary)
    summary_df = summary_df.sort_values('selected_conflicts', ascending=False)
    
    # Auto-generate output path if not provided
    if args.repository_summary_csv:
        summary_output_path = Path(args.repository_summary_csv)
    else:
        summary_output_path = Path(args.csv_out).parent / "repository_summary.csv"
    
    summary_df.to_csv(summary_output_path, index=False, encoding="utf-8")
    
    # Log summary statistics
    total_repos = len(summary_df)
    repos_with_conflicts = len(summary_df[summary_df['selected_conflicts'] > 0])
    total_selected = summary_df['selected_conflicts'].sum()
    
    logger.info(f"Repository summary written to {summary_output_path}")
    logger.info(f"Total repositories with conflicts: {total_repos}")
    logger.info(f"Repositories contributing to final dataset: {repos_with_conflicts}")
    logger.info(f"Total selected conflicts across all repositories: {total_selected}")


def create_basic_repository_summary(df: pd.DataFrame, args) -> None:
    """Create a basic summary without repository mapping."""
    total_conflicts = len(df)
    selected_conflicts = df['selected'].sum()
    
    basic_summary = [{
        'repository': 'ALL_REPOSITORIES',
        'total_conflicts': total_conflicts,
        'selected_conflicts': selected_conflicts,
        'filtered_out_conflicts': total_conflicts - selected_conflicts,
        'failed_max_lines': df['fail_max_lines'].sum(),
        'failed_token_count': df['fail_token_count'].sum(),
        'failed_incoherent': df['fail_incoherent'].sum(),
        'selection_rate': selected_conflicts / total_conflicts if total_conflicts > 0 else 0.0
    }]
    
    summary_df = pd.DataFrame(basic_summary)
    summary_output_path = Path(args.csv_out).parent / "repository_summary.csv"
    summary_df.to_csv(summary_output_path, index=False, encoding="utf-8")
    
    logger.info(f"Basic repository summary written to {summary_output_path}")


def main():  # pylint: disable=too-many-statements, too-many-locals
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Compute metrics for each conflict snippet "
        "(context, conflict, resolution, and parent versions), "
        "and optionally filter/copy selected conflicts."
    )
    parser.add_argument(
        "--input_dir",
        default="merges/repos_50/conflict_blocks",
        help="Directory containing <basename><n>.conflict and "
        "<basename><n>.resolved_conflict files",
    )
    parser.add_argument(
        "--csv_out",
        default="merges/repos_50/conflict_metrics.csv",
        help="Path to the output CSV file.",
    )
    parser.add_argument(
        "--max_line_count",
        type=int,
        default=20,
        help="Maximum number of lines allowed in the entire .conflict file. "
        "Conflicts exceeding this will be marked not filtered.",
    )
    parser.add_argument(
        "--filtered_output_dir",
        type=str,
        default="merges/repos_50/filtered_conflicts",
        help="If specified, copy filtered .conflict/.resolved_conflict pairs to this directory.",
    )
    parser.add_argument(
        "-keep_trivial_resolution",
        action="store_true",
        help="Filter out conflict blocks with trivial resolutions.",
    )
    parser.add_argument(
        "--conflict_files_csv",
        type=str,
        help="Path to conflict_files.csv to map merge IDs to repositories (auto-detected if not provided)",
    )
    parser.add_argument(
        "--repository_summary_csv",
        type=str,
        help="Path to output repository summary CSV (auto-generated if not provided)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        logger.error(f"ERROR: input_dir '{input_dir}' is not a directory.")
        sys.exit(1)

    # Collect all *.conflict files; we'll match each with a corresponding *.resolved_conflict
    conflict_files = sorted(input_dir.glob("*.conflict"))
    if not conflict_files:
        logger.info("No '.conflict' files found.")
        sys.exit(0)

    # Create output directory if needed
    output_dir = Path(args.filtered_output_dir)
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare a list for metric rows
    rows = []
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    for conflict_path in tqdm(conflict_files):
        # For each .conflict, find the corresponding .resolved_conflict file
        resolved_path = conflict_path.with_suffix(".resolved_conflict")
        if not resolved_path.exists():
            logger.info(f"No matching .resolved_conflict for {conflict_path}. Skipped.")
            continue

        # Use the filename (minus extension) as the conflict identifier
        identifier = conflict_path.stem  # e.g. "myfile1"

        # Read lines (without newline characters)
        conflict_lines = conflict_path.read_text(encoding="utf-8").splitlines()
        resolved_lines = resolved_path.read_text(encoding="utf-8").splitlines()

        try:
            # Split the conflict snippet into context and conflict block
            before_ctx, conflict_block, after_ctx = split_single_conflict_snippet(
                conflict_lines
            )
        except ValueError as e:
            logger.info(f"{e} in {conflict_path}. Skipped.")
            continue

        # Further split the conflict block into left_parent, right_parent, and base (if available)
        left_parent, right_parent, base = split_conflict_block(conflict_block)

        # Derive the resolved conflict block by removing the context portions
        # from the resolved file.
        if len(after_ctx) < 1:
            resolution = resolved_lines[len(before_ctx) :]
        else:
            resolution = resolved_lines[len(before_ctx) : -len(after_ctx)]

        # Check if the resolution is fully contained in either left_parent or right_parent.
        res_in_left = functional_equality(resolution, left_parent)
        res_in_right = functional_equality(resolution, right_parent)

        # Check if the resolution is incoherent (i.e., it is larger than the conflict block)
        incoherent_resolution_size = len(conflict_lines) + 2 < len(resolved_lines)

        # Compute all metrics
        metrics = {
            "full_conflict_lines": len(conflict_lines),
            "context_before_lines": len(before_ctx),
            "conflict_lines": len(conflict_block),
            "context_after_lines": len(after_ctx),
            "full_resolution_lines": len(resolved_lines),
            "resolution_lines": len(resolution),
            "parent1_lines": len(left_parent),
            "parent2_lines": len(right_parent),
            "base_lines": len(base) if base is not None else -1,
            "num_tokens_query": get_token_count(tokenizer, conflict_lines),
            "resolution_in_left_or_right": (res_in_left or res_in_right),
            "incoherent_resolution_size": incoherent_resolution_size,
        }

        # Decide if this conflict should be selected based on filtering rules
        selected = True
        row_data = {"conflict_id": identifier}
        row_data["fail_max_lines"] = (
            metrics["full_conflict_lines"] > args.max_line_count
        )
        row_data["fail_token_count"] = metrics["num_tokens_query"] > MAX_PROMPT_LENGTH
        row_data["fail_incoherent"] = incoherent_resolution_size

        selected = not (
            row_data["fail_max_lines"]
            or row_data["fail_token_count"]
            or row_data["fail_incoherent"]
        )

        # If selected, copy the conflict and resolved files to the output directory
        if selected:
            shutil.copy(conflict_path, output_dir / conflict_path.name)
            shutil.copy(resolved_path, output_dir / resolved_path.name)

        # Prepare row data
        row_data.update(metrics)  # type: ignore
        row_data["selected"] = selected  # type: ignore
        rows.append(row_data)

    total_merges = len(rows)
    if total_merges > 0:
        percentage_max_lines = (
            sum(row["fail_max_lines"] for row in rows) / total_merges * 100
        )
        percentage_token = (
            sum(row["fail_token_count"] for row in rows) / total_merges * 100
        )
        percentage_incoherent = (
            sum(row["fail_incoherent"] for row in rows) / total_merges * 100
        )
        logger.info(
            f"Percentage of merges failing due to max_line_count: {percentage_max_lines:.2f}%"
        )
        logger.info(
            "Percentage of merges failing due to token "
            f"count exceeding prompt limit: {percentage_token:.2f}%"
        )
        logger.info(
            "Percentage of merges failing due to incoherent "
            f"resolution: {percentage_incoherent:.2f}%"
        )

    # Create a pandas DataFrame from the list of rows and write it to CSV
    df = pd.DataFrame(rows)
    df.to_csv(args.csv_out, index=False, encoding="utf-8")

    logger.info(f"Metrics (with 'selected' column) have been written to {args.csv_out}")
    logger.info(f"Selected conflicts copied to {output_dir}")
    logger.info(f"Number of selected conflicts: {df['selected'].sum()}")

    # Generate repository summary
    generate_repository_summary(df, args, input_dir)


if __name__ == "__main__":
    main()
