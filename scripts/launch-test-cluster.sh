#!/bin/bash

# Usage: ./launch-test-cluster.sh [options]
# Options:
#   -c CLUSTER_NAME    Set cluster name (default: art-integration-test)
#   --no-pull          Skip git pull before launching
#   -n NOTEBOOKS       Comma-separated notebook indexes to run (e.g., '0,2' or '1-3')
#
# Examples:
#   ./launch-test-cluster.sh --no-pull          # Run all tests
#   ./launch-test-cluster.sh -n 0 --no-pull     # Run only notebook 0
#   ./launch-test-cluster.sh -n 0,2 --no-pull   # Run notebooks 0 and 2
#   ./launch-test-cluster.sh -c my-cluster -n 1-3 --no-pull  # Custom cluster, run notebooks 1-3

CLUSTER_NAME="art-integration-test"

# Parse arguments
LAUNCH_ARGS=()
NOTEBOOKS=""
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
    -n)
      NOTEBOOKS="$2"
      shift 2
      ;;
    *)
      LAUNCH_ARGS+=("$1")
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
  LAUNCH_ARGS+=(--env "GIT_RESET_CLEAN=false")
fi

echo "Launching cluster \"$CLUSTER_NAME\"..."
uv run sky launch skypilot-config.yaml -c "$CLUSTER_NAME" --env-file .env -y "${LAUNCH_ARGS[@]}"
LAUNCH_EXIT=$?
if [[ $LAUNCH_EXIT -ne 0 ]]; then
  echo "Error: Cluster launch failed with exit code $LAUNCH_EXIT"
  exit $LAUNCH_EXIT
fi

echo "Running tests on \"$CLUSTER_NAME\"..."
if [[ -n "$NOTEBOOKS" ]]; then
  echo "Running notebooks: $NOTEBOOKS"
  TEST_CMD="uv run tests/integration.py -n '$NOTEBOOKS'"
else
  TEST_CMD="uv run tests/integration.py"
fi

uv run sky exec -c "$CLUSTER_NAME" --env-file .env \
  "bash -lc 'export CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-0}; $TEST_CMD || true; pkill -f python || true; exit'"
TEST_EXIT_CODE=$?

echo "Test completed."

# Exit with the test command's exit code; trap will down the cluster.
exit $TEST_EXIT_CODE