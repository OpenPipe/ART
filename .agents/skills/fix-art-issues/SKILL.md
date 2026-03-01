---
name: fix-art-issues
description: >
  Fix a GitHub issue on OpenPipe/ART and open a PR.
  Use when the user asks to fix, solve, or work on an ART issue,
  or references a GitHub issue URL containing "OpenPipe/ART".
  Triggers: "fix ART issue", "solve this issue" with an OpenPipe/ART URL,
  "work on ART #N".
---

# Fix ART Issue

Fix a GitHub issue on `OpenPipe/ART` and open a PR.

- **Repo**: `OpenPipe/ART`
- **Base branch**: `main`

Assumes the workspace is already set up with the correct branch checked out and `.env` in place (handled by the system-level `fix-art-workspace` skill).

## Workflow

### 1. Read the Issue
```
gh issue view <number> --repo OpenPipe/ART --json title,body,labels,assignees,comments
```

### 2. Explore, Plan, Implement
- Use the Explore agent to understand relevant code before making changes.
- Plan clearly, implement with minimal focused changes. No over-engineering.

### 3. Commit and Push
- Commit with a message that includes `Closes #<issue-number>`.
- Push the feature branch. If HTTPS push fails due to SAML SSO, set SSH remote: `git remote set-url origin git@github.com:OpenPipe/ART.git`

### 4. Open a Draft PR
- `gh pr create --base main --draft`.
- PR body: `## Summary`, `Closes #<number>`, `## Changes`, `## Test plan`.

### 5. Testing
- **No test artifacts in the final PR**: debug prints, test scripts, and temporary changes must NOT be committed.
- Update the PR's test plan section with detailed results.
- When testing passes, mark the PR as ready: `gh pr ready`.

## Reference

Read `CONTRIBUTING.md` at the repo root for guidance on code quality checks (prek), CI cache refresh, and the release process.

## Dependency Management Tips

- **Pin versions strictly** (`==`) for critical deps like `transformers`, `trl`, `unsloth`, `unsloth-zoo`, `vllm` to avoid surprise breakage from new releases.
- **Don't loosen pins without reason**: if a dep was `==X.Y.Z`, keep it pinned unless there's a specific reason to change. Don't use `>=` just because it seems more flexible.
- **`uv run` fails on macOS** for backend deps (apex/torch need CUDA). This is expected — use `uvx ruff` for linting locally, test on GPU cluster.

## Deploying a GPU Cluster

Name the SkyPilot cluster after the branch name without the `fix/` prefix, replacing `/` with `-` (SkyPilot doesn't allow slashes). For example, if the branch is `fix/short-description`:
```
uv run sky launch -c short-description skypilot-config.yaml -y
```

To connect: `ssh short-description`

To tear down when done: `uv run sky down short-description`

## GPU Cluster Testing Tips

- **Kill stale GPU processes** before re-running tests: `nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9`. Previous failed runs leave processes holding GPU memory.
- **Set `gpu_memory_utilization`** in test scripts (e.g. `0.7`) — the default `0.9` is too high when Unsloth's training model is also loaded on the same GPU.
- **Redirect test output to a log file**: `nohup python test.py > /tmp/output.log 2>&1 &` then `tail -f /tmp/output.log`. SSH background tasks lose output when connection drops.
- **Git on cluster**: SSH keys may not be configured. Use HTTPS with token: `git remote set-url origin https://${GITHUB_TOKEN}@github.com/OpenPipe/ART.git`
- **Tear down clusters** when done: `sky down <cluster-name> -y`

$ARGUMENTS
