#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

IMAGE_REPO="${IMAGE_REPO:-ghcr.io/openpipe/art-ci}"
BASE_IMAGE="${BASE_IMAGE:-pytorch/pytorch:2.9.0-cuda12.8-cudnn9-devel}"
PYTHON_MM="${PYTHON_MM:-3.11}"
BUILD_JOBS="${BUILD_JOBS:-auto}"
WORKFLOW_FILE="${WORKFLOW_FILE:-build-prek-cache-image.yml}"
WORKFLOW_SELECTOR=""
REF=""
WATCH=0

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/ci/build_and_push_prek_image.sh [options]

Description:
  Dispatches the GitHub Actions workflow that builds and pushes the prebuilt
  Prek cache image to GHCR. No local docker/podman setup is required.

Options:
  --image-repo <repo>      Image repository (default: ghcr.io/openpipe/art-ci)
  --base-image <image>     Base image used by the cache-prewarm Dockerfile
  --python-mm <mm>         Python major.minor used in CI (default: 3.11)
  --build-jobs <n|auto>    Parallel native-build jobs in image build (default: auto)
  --ref <git-ref>          Git ref for workflow_dispatch (default: current branch)
  --workflow <file>        Workflow selector (name or path, default: build-prek-cache-image.yml)
  --watch                  Watch the dispatched workflow run and return its status
  -h, --help               Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image-repo)
      IMAGE_REPO="$2"
      shift 2
      ;;
    --base-image)
      BASE_IMAGE="$2"
      shift 2
      ;;
    --python-mm)
      PYTHON_MM="$2"
      shift 2
      ;;
    --build-jobs)
      BUILD_JOBS="$2"
      shift 2
      ;;
    --ref)
      REF="$2"
      shift 2
      ;;
    --workflow)
      WORKFLOW_FILE="$2"
      shift 2
      ;;
    --watch)
      WATCH=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

WORKFLOW_SELECTOR="${WORKFLOW_FILE}"

log() {
  printf '[ci-image] %s\n' "$*"
}

fail() {
  printf '[ci-image] ERROR: %s\n' "$*" >&2
  exit 1
}

require_cmd() {
  local cmd="$1"
  command -v "${cmd}" >/dev/null 2>&1 || fail "Required command not found: ${cmd}"
}

compute_fingerprint() {
  python3 "${REPO_ROOT}/scripts/ci/compute_megatron_fingerprint.py" \
    --pyproject "${REPO_ROOT}/pyproject.toml" \
    --base-image "${BASE_IMAGE}" \
    --python-mm "${PYTHON_MM}"
}

resolve_ref() {
  if [[ -n "${REF}" ]]; then
    printf '%s\n' "${REF}"
    return
  fi

  local branch
  branch="$(git -C "${REPO_ROOT}" rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
  if [[ -z "${branch}" || "${branch}" == "HEAD" ]]; then
    branch="$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || true)"
  fi
  [[ -n "${branch}" ]] || fail "Could not determine git ref. Pass --ref explicitly."
  printf '%s\n' "${branch}"
}

dispatch_once() {
  local selector="$1"
  local ref="$2"
  local fingerprint="$3"
  gh workflow run "${selector}" \
    --ref "${ref}" \
    -f image_repo="${IMAGE_REPO}" \
    -f base_image="${BASE_IMAGE}" \
    -f python_mm="${PYTHON_MM}" \
    -f build_jobs="${BUILD_JOBS}" \
    -f fingerprint="${fingerprint}"
}

dispatch_workflow() {
  local ref="$1"
  local fingerprint="$2"
  local output=""
  local fallback_selector=""

  if [[ "${WORKFLOW_SELECTOR}" != *"/"* ]]; then
    fallback_selector=".github/workflows/${WORKFLOW_SELECTOR}"
  fi

  log "Dispatching workflow ${WORKFLOW_SELECTOR} on ref ${ref}."
  if output="$(dispatch_once "${WORKFLOW_SELECTOR}" "${ref}" "${fingerprint}" 2>&1)"; then
    printf '%s\n' "${output}"
    return
  fi

  if [[ -n "${fallback_selector}" && "${fallback_selector}" != "${WORKFLOW_SELECTOR}" ]]; then
    log "Workflow selector ${WORKFLOW_SELECTOR} was not found; retrying ${fallback_selector}."
    if output="$(dispatch_once "${fallback_selector}" "${ref}" "${fingerprint}" 2>&1)"; then
      WORKFLOW_SELECTOR="${fallback_selector}"
      printf '%s\n' "${output}"
      return
    fi
  fi

  fail "Could not dispatch workflow '${WORKFLOW_FILE}'. Push the workflow file to GitHub first. If this is a new workflow, GitHub may require it on the default branch before workflow_dispatch. gh output: ${output}"
}

watch_latest_run() {
  local run_id
  sleep 2
  run_id="$(gh run list --workflow "${WORKFLOW_SELECTOR}" --event workflow_dispatch --limit 1 --json databaseId --jq '.[0].databaseId' 2>/dev/null || true)"
  [[ -n "${run_id}" && "${run_id}" != "null" ]] || fail "Could not find dispatched run. Check with: gh run list --workflow ${WORKFLOW_SELECTOR}"

  log "Watching workflow run ${run_id}."
  gh run watch "${run_id}" --exit-status
}

main() {
  require_cmd gh
  require_cmd git
  require_cmd python3

  gh auth status >/dev/null 2>&1 || fail "GitHub auth not configured. Run: gh auth login"

  local resolved_ref
  resolved_ref="$(resolve_ref)"

  local fingerprint
  fingerprint="$(compute_fingerprint)"

  local immutable_tag="${IMAGE_REPO}:prek-megatron-${fingerprint}"
  local current_tag="${IMAGE_REPO}:prek-megatron-current"

  dispatch_workflow "${resolved_ref}" "${fingerprint}"

  cat <<MSG
[ci-image] Dispatched build workflow.
[ci-image] Fingerprint: ${fingerprint}
[ci-image] Immutable tag: ${immutable_tag}
[ci-image] Current tag:   ${current_tag}
[ci-image] Build jobs:    ${BUILD_JOBS}
[ci-image] Follow progress with:
[ci-image]   gh run list --workflow ${WORKFLOW_SELECTOR} --limit 5
MSG

  if [[ "${WATCH}" -eq 1 ]]; then
    watch_latest_run
  fi
}

main
