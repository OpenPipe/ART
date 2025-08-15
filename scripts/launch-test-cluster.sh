#!/bin/bash

CLUSTER_NAME="art-integration-test"

# Parse arguments
ARGS=()
PULL_LATEST=true
while [[ $# -gt 0 ]]; do
  case "$1" in
    -c)
      CLUSTER_NAME="$2"
      shift 2
      ;;
    --no-pull)
      PULL_LATEST=false
      shift 1
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

# Always attempt to tear down the cluster on exit
trap 'echo "Tearing down cluster \"$CLUSTER_NAME\"..."; uv run sky down -y "$CLUSTER_NAME" || true' EXIT

# Check for unstaged changes
if ! git diff --quiet; then
  echo "Warning: You have unstaged changes. Unstaged changes will be discarded from the cluster working directory."
fi

# Check for uncommitted changes
if ! git diff --cached --quiet; then
  echo "Warning: You have uncommitted changes. Uncommitted changes will be discarded from the cluster working directory."
fi

if [[ "$PULL_LATEST" == true ]]; then
  echo "Pulling latest changes..."
  if ! git pull; then
    echo "Error: Failed to pull latest changes."
    exit 1
  fi
else
  echo "Skipping git pull (deploying current working tree). To pull latest, omit --no-pull."
  # Preserve synced working tree on remote by disabling reset/clean.
  ARGS+=(--env "GIT_RESET_CLEAN=false")
fi

echo "Launching cluster \"$CLUSTER_NAME\"..."
uv run sky launch skypilot-config.yaml -c "$CLUSTER_NAME" --env-file .env -y "${ARGS[@]}"
LAUNCH_EXIT=$?
if [[ $LAUNCH_EXIT -ne 0 ]]; then
  echo "Error: Cluster launch failed with exit code $LAUNCH_EXIT"
  exit $LAUNCH_EXIT
fi

echo "Running tests on \"$CLUSTER_NAME\"..."
uv run sky exec -c "$CLUSTER_NAME" --env-file .env \
  "bash -lc 'export CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-0}; uv run tests/integration.py || true; pkill -f python || true; exit'"
TEST_EXIT_CODE=$?

echo "Test completed."

# Exit with the test command's exit code; trap will down the cluster.
exit $TEST_EXIT_CODE