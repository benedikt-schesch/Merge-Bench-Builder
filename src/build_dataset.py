# -*- coding: utf-8 -*-
"""Builds a dataset from the conflict blocks."""

import os
import argparse
import random
from pathlib import Path
from typing import Dict, Union, List
import numpy as np
from datasets import Dataset, DatasetDict
from tqdm import tqdm
from loguru import logger
import torch
from variables import QUERY_PROMPT

logger.add("run.log", backtrace=True, diagnose=True)


def build_query(query: str, language: str = "java") -> str:
    """Builds a query from the given conflict block."""
    return f"{QUERY_PROMPT}```{language}\n{query}\n```"


def load_conflict_dataset(directory: str, language: str = "java") -> Dataset:
    """Loads the conflict dataset from the given directory."""
    # Sort by name deterministically to ensure reproducible dataset creation
    conflict_files = sorted(Path(directory).glob("*.conflict"))
    queries, solutions = [], []

    for conflict_file in tqdm(conflict_files):
        resolved_file = conflict_file.with_name(
            conflict_file.stem + ".resolved_conflict"
        )
        queries.append(build_query(conflict_file.read_text(encoding="utf-8"), language))
        solutions.append(resolved_file.read_text(encoding="utf-8"))

    if not queries:
        raise ValueError("No valid conflict/solution pairs found.")
    return Dataset.from_dict({"question": queries, "answer": solutions})


def format_conversation(
    example: Dict[str, str],
) -> Dict[str, Union[str, List[Dict[str, str]]]]:
    """Formats the conversation for the chatbot model."""
    return {
        "prompt": [
            {"role": "user", "content": example["question"]},
        ],
        "answer": example["answer"],
    }


def prepare_dataset(
    directory: str, test_size: float = 0.2, seed: int = 42, language: str = "java"
) -> DatasetDict:
    """Prepare the dataset for training and testing."""
    dataset = load_conflict_dataset(directory, language)

    # Handle edge cases for test_size 0 or 1
    if not test_size:
        # All data in train, test empty
        dataset = dataset.map(format_conversation)
        empty_test = dataset.select([])
        return DatasetDict({"train": dataset, "test": empty_test})
    if test_size == 1:
        # All data in test, train empty
        dataset = dataset.map(format_conversation)
        empty_train = dataset.select([])
        return DatasetDict({"train": empty_train, "test": dataset})
    splits = dataset.train_test_split(test_size=test_size, seed=seed)
    splits["train"] = splits["train"].map(format_conversation)
    splits["test"] = splits["test"].map(format_conversation)
    return DatasetDict({"train": splits["train"], "test": splits["test"]})


def main():
    """Main function for building the dataset."""
    parser = argparse.ArgumentParser(
        description="Prepare train/test dataset from conflict blocks."
    )
    parser.add_argument(
        "--conflict_blocks_dir",
        type=str,
        default="merges/repos_reaper_1000/filtered_dataset",
    )
    parser.add_argument(
        "--output_dir", type=str, default="merges/repos_reaper_1000/dataset"
    )
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="java",
        help="Programming language for syntax highlighting",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.manual_seed(args.seed)

    dataset = prepare_dataset(
        args.conflict_blocks_dir, args.test_size, args.seed, args.language
    )
    logger.info(f"Train set size: {len(dataset['train'])}")
    logger.info(f"Test set size: {len(dataset['test'])}")

    os.makedirs(args.output_dir, exist_ok=True)
    dataset.save_to_disk(args.output_dir)
    logger.info(f"Dataset saved to {args.output_dir}")


if __name__ == "__main__":
    main()
