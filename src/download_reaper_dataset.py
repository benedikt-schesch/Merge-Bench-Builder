#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Download repo list."""

# usage: python3 get_repos.py

# This script creates file input_data/repos.csv from part of the Reaper dataset:
# with problematic repositories removed.
# Language and star filtering is now done in the sampling step.
# This script only needs to be re-run when you desire to re-create that file (which is rare).

import gzip
import urllib.request
from io import BytesIO

import pandas as pd
import numpy as np
from loguru import logger

repos_csv = "input_data/repos_reaper.csv"

if __name__ == "__main__":
    urllib.request.urlretrieve(
        "https://reporeapers.github.io/static/downloads/dataset.csv.gz",
        "input_data/repos.csv.gz",
    )
    with gzip.open("input_data/repos.csv.gz", "rb") as f:
        df = pd.read_csv(BytesIO(f.read()))
    
    # Print language statistics
    language_counts = df['language'].value_counts()
    logger.info("Repository count by language:")
    for language, count in language_counts.head(15).items():  # Show top 15 languages
        logger.info(f"  {language}: {count}")
    logger.info(f"Total languages: {len(language_counts)}")
    logger.info(f"Total repositories before filtering: {len(df)}")
    
    df = df.replace(to_replace="None", value=np.nan).dropna()
    df["stars"] = df["stars"].astype(int)

    # Remove specific repositories that had difficulties in their testing environments
    problematic_repos = [
        "elastic/elasticsearch",
        "wasabeef/RecyclerViewAnimators", 
        "android/platform_frameworks_base",
        "elasticsearch/elasticsearch"
    ]
    df = df[~df["repository"].isin(problematic_repos)]

    df.to_csv(repos_csv, index_label="idx")

    logger.info("Number of repos written to " + repos_csv + " : " + str(len(df)))
