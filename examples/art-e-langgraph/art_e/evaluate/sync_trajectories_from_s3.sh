#! /bin/bash

# Change to the repository root (relative to the script location)
cd "$(dirname "$0")/../.." || exit 1

aws s3 sync s3://email-deep-research-backups/email_agent/models/ .art/email_agent/models/ \
  --exclude "*" \
  --include "*/trajectories/" \
  --include "*/trajectories/*"