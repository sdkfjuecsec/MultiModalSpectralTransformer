#!/bin/bash
#
# Commit every local change and push it to origin.
#
# Purpose
#   Mirror the development working tree to the remote so the other environment
#   (training cluster / laptop) can `bash pull_latest.sh` it.
#
# Usage
#   bash push_changes.sh "Your commit message"
#   The commit message is mandatory; a missing argument exits with status 1.
#
# Note
#   `git add .` stages *everything* that is not ignored, so keep large data
#   files, checkpoints and any credentials in .gitignore before running this.
#

# Script to commit and push changes from the development environment
# Usage: bash push_changes.sh "Your commit message"

# Check if a commit message was provided
if [ -z "$1" ]; then
  echo "Error: Please provide a commit message."
  echo "Usage: bash push_changes.sh \"Your commit message\""
  exit 1
fi

# Get current branch
CURRENT_BRANCH=$(git branch --show-current)
echo "Current branch: $CURRENT_BRANCH"

# Add all changes
echo "Adding all changes..."
git add .

# Commit changes with the provided message
echo "Committing changes with message: $1"
git commit -m "$1"

# Push changes to the remote repository
echo "Pushing changes to origin/$CURRENT_BRANCH..."
git push origin $CURRENT_BRANCH

echo "Done! Changes have been pushed to origin/$CURRENT_BRANCH."
