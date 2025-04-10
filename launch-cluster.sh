#!/bin/bash

CLUSTER_NAME="art"

# Parse arguments
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -c)
      CLUSTER_NAME="$2"
      shift 2
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

# Check for unstaged changes
if ! git diff --quiet; then
    echo "Warning: You have unstaged changes. Unstaged changes will be discarded from the cluster working directory."
fi

# Check for uncommitted changes
if ! git diff --cached --quiet; then
    echo "Warning: You have uncommitted changes. Uncommitted changes will be discarded from the cluster working directory."
fi

# Pull latest changes
echo "Pulling latest changes..."
if ! git pull; then
    echo "Error: Failed to pull latest changes."
    exit 1
fi

# Launch the cluster
uv run sky launch skypilot-config.yaml -c "$CLUSTER_NAME" --env-file .env -y "${ARGS[@]}"