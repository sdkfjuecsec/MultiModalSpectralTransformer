#!/bin/bash
#
# Sync the local checkout with the remote `main_cleaned` branch.
#
# Purpose
#   Throw away every local modification and end up exactly at the tip of
#   origin/main_cleaned.
#
# Usage
#   bash pull_latest.sh          (no arguments, run from the repo root)
#
# Warning
#   The first step is `git reset --hard`, so uncommitted work is lost. Commit or
#   stash before running this.
#

# Script to reset to the remote branch and pull the latest updates
# Hard coded to use origin/main_cleaned

# NOTE (known issue, left unchanged): the order is reset -> fetch -> pull, so the
# reset targets the *last fetched* origin/main_cleaned, not the current remote
# tip. On a stale checkout this lands on an older commit and the following
# `git pull` then has to merge/replay the missing history; running fetch first
# would make the reset land on the true tip.

# Set target branch to main_cleaned
TARGET_BRANCH="main_cleaned"
echo "Resetting to origin/$TARGET_BRANCH..."

# Reset hard to the remote branch
git reset --hard origin/$TARGET_BRANCH

# Fetch all branches from remote
echo "Fetching updates from remote..."
git fetch --all

# Pull the latest changes to ensure we're up to date
echo "Pulling latest changes from origin/$TARGET_BRANCH..."
git pull origin $TARGET_BRANCH

echo "Done! Local repository has been reset to origin/$TARGET_BRANCH."
 