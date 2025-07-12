#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Download top repositories from GitHub API for various languages."""

import os
import time
import requests
import pandas as pd
from loguru import logger
from typing import List, Dict, Optional
import argparse

class GitHubRepoFetcher:
    """Fetches top repositories from GitHub API."""
    
    def __init__(self, token: Optional[str] = None):
        """Initialize with optional GitHub token for better rate limits."""
        self.token = token
        self.headers = {}
        if token:
            self.headers['Authorization'] = f'token {token}'
        self.base_url = 'https://api.github.com'
        
    def search_repositories(self, query: str, sort: str = 'stars', 
                          order: str = 'desc', per_page: int = 100, 
                          max_results: int = 1000) -> List[Dict]:
        """Search repositories using GitHub API."""
        repos = []
        page = 1
        
        # GitHub API limits search results to 1000
        max_results = min(max_results, 1000)
        
        while len(repos) < max_results:
            url = f"{self.base_url}/search/repositories"
            params = {
                'q': query,
                'sort': sort,
                'order': order,
                'per_page': per_page,
                'page': page
            }
            
            logger.info(f"Fetching page {page} (current total: {len(repos)} repos)")
            
            try:
                response = requests.get(url, headers=self.headers, params=params)
                response.raise_for_status()
                
                # Check rate limit
                remaining = int(response.headers.get('X-RateLimit-Remaining', 0))
                if remaining < 10:
                    reset_time = int(response.headers.get('X-RateLimit-Reset', 0))
                    sleep_time = max(reset_time - time.time(), 0) + 1
                    logger.warning(f"Rate limit low ({remaining} remaining). Sleeping for {sleep_time} seconds...")
                    time.sleep(sleep_time)
                
                data = response.json()
                items = data.get('items', [])
                
                if not items:
                    logger.info("No more repositories found")
                    break
                
                repos.extend(items)
                
                # Check if we've reached the total count
                total_count = data.get('total_count', 0)
                if len(repos) >= total_count:
                    logger.info(f"Fetched all available repos (total: {total_count})")
                    break
                
                page += 1
                
                # Small delay to be respectful to API
                time.sleep(0.5)
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Error fetching repositories: {e}")
                break
        
        # Trim to max_results
        repos = repos[:max_results]
        logger.info(f"Fetched {len(repos)} repositories")
        
        return repos
    
    def format_repo_data(self, repos: List[Dict]) -> pd.DataFrame:
        """Format GitHub API data to match Reaper dataset structure."""
        formatted_data = []
        
        for repo in repos:
            # Extract relevant fields
            formatted_repo = {
                'repository': repo['full_name'],
                'stars': repo['stargazers_count'],
                'language': repo.get('language', ''),
                'created_at': repo['created_at'],
                'updated_at': repo['updated_at'],
                'forks': repo['forks_count'],
                'open_issues': repo['open_issues_count'],
                'size': repo['size'],
                'default_branch': repo.get('default_branch', 'main'),
                'description': repo.get('description', ''),
                'url': repo['html_url'],
                'clone_url': repo['clone_url'],
                'archived': repo.get('archived', False),
                'disabled': repo.get('disabled', False),
                # Add placeholder values for Reaper-specific fields
                'architecture': 1.0,  # Placeholder
                'community': 1.0,     # Placeholder
                'continuous_integration': 1.0,  # Placeholder
                'documentation': 1.0,  # Placeholder
                'history': 1.0,       # Placeholder
                'issues': 1.0,        # Placeholder
                'license': 1.0 if repo.get('license') else 0.0,
                'size_metric': 1.0,   # Placeholder
                'unit_test': 1.0      # Placeholder
            }
            formatted_data.append(formatted_repo)
        
        return pd.DataFrame(formatted_data)


def main():
    """Main function to download repositories."""
    parser = argparse.ArgumentParser(description='Download top repositories from GitHub')
    parser.add_argument('--token', type=str, help='GitHub personal access token', 
                       default=os.environ.get('GITHUB_TOKEN'))
    parser.add_argument('--language', type=str, default='javascript',
                       help='Programming language to search for')
    parser.add_argument('--max-results', type=int, default=1000,
                       help='Maximum number of repositories to fetch (max 1000 due to API limit)')
    parser.add_argument('--min-stars', type=int, default=100,
                       help='Minimum number of stars')
    parser.add_argument('--output', type=str, 
                       default=None,
                       help='Output CSV file path')
    parser.add_argument('--exclude-archived', action='store_true',
                       help='Exclude archived repositories')
    
    args = parser.parse_args()
    
    # Build search query
    query_parts = [f'language:{args.language}']
    if args.min_stars:
        query_parts.append(f'stars:>={args.min_stars}')
    if args.exclude_archived:
        query_parts.append('archived:false')
    
    query = ' '.join(query_parts)
    logger.info(f"Search query: {query}")
    
    # Initialize fetcher
    fetcher = GitHubRepoFetcher(token=args.token)
    
    if not args.token:
        logger.warning("No GitHub token provided. API rate limits will be restrictive (60 requests/hour).")
        logger.warning("Set GITHUB_TOKEN environment variable or use --token parameter for better limits.")
    
    # Fetch repositories
    logger.info(f"Fetching top {args.max_results} {args.language} repositories...")
    repos = fetcher.search_repositories(query, max_results=args.max_results)
    
    if not repos:
        logger.error("No repositories fetched. Check your connection and API limits.")
        return
    
    # Format and save data
    df = fetcher.format_repo_data(repos)
    
    # Remove truly disabled/archived repos if requested
    if args.exclude_archived:
        initial_count = len(df)
        df = df[(df['archived'] == False) & (df['disabled'] == False)]
        logger.info(f"Filtered out {initial_count - len(df)} archived/disabled repositories")
    
    # Sort by stars descending
    df = df.sort_values(by='stars', ascending=False)
    
    # Generate default output path if not specified
    if args.output is None:
        output_path = f"input_data/repos_github_{args.language.capitalize()}_0_{len(df)}.csv"
    else:
        output_path = args.output
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df)} repositories to {output_path}")
    
    # Print summary statistics
    logger.info("\nRepository Statistics:")
    logger.info(f"  Total repositories: {len(df)}")
    logger.info(f"  Stars range: {df['stars'].min()} - {df['stars'].max()}")
    logger.info(f"  Average stars: {df['stars'].mean():.0f}")
    logger.info(f"  Median stars: {df['stars'].median():.0f}")
    
    # Show top 10 repositories
    logger.info("\nTop 10 repositories by stars:")
    for idx, row in df.head(10).iterrows():
        logger.info(f"  {row['repository']}: {row['stars']} stars")


if __name__ == "__main__":
    main()
