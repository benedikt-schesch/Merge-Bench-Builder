# -*- coding: utf-8 -*-
"""Sample script to filter top N repositories by stars."""

import argparse
import os
import pandas as pd


def filter_top_n_repos(input_path, start_index, n, language):
    """Filter repositories starting at a given index and select n rows."""
    # Read the CSV file
    df = pd.read_csv(input_path)

    # Filter for specified language repositories
    df = df[df["language"] == language]

    # Ensure 'stars' column is numeric
    df["stars"] = pd.to_numeric(df["stars"], errors="coerce")

    # Drop rows with NaN values in 'stars'
    df = df.dropna(subset=["stars"])

    # Sort by 'stars' in descending order
    df_sorted = df.sort_values(by="stars", ascending=False)

    # Select n rows starting at the specified index
    df_top_n = df_sorted.iloc[start_index : start_index + n]

    # Output file name
    output_path = f"input_data/repos_reaper_{language}_{start_index}_{start_index + n}.csv"

    # Save to new CSV file
    df_top_n.to_csv(output_path, index=False)
    print(f"Top {n} {language} repositories saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter top N repositories by stars.")
    parser.add_argument(
        "--n", type=int, default=100, help="Number of top repositories to select."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default="input_data/repos_reaper.csv",
        help="Path to the input CSV file.",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Starting index for selecting repositories.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="Java",
        help="Programming language to filter repositories by (default: Java).",
    )
    args = parser.parse_args()

    if os.path.exists(args.input_path):
        filter_top_n_repos(args.input_path, args.start_index, args.n, args.language)
    else:
        print(f"Error: The file {args.input_path} does not exist.")
