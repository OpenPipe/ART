#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

BASE_IMAGE="${BASE_IMAGE:-pytorch/pytorch:2.9.0-cuda12.8-cudnn9-devel}"
PYTHON_MM="${PYTHON_MM:-3.11}"
UV_CACHE_RELEASE_TAG="${UV_CACHE_RELEASE_TAG:-prek-uv-cache}"
UV_CACHE_ASSET_PREFIX="${UV_CACHE_ASSET_PREFIX:-prek-uv-cache}"
BUILD_JOBS="${BUILD_JOBS:-auto}"
KEEP_COUNT="${KEEP_COUNT:-4}"
PART_SIZE_MB="${PART_SIZE_MB:-1900}"
SKIP_BUILD=0
SKIP_PRUNE=0
ARCHIVE_PATH=""
TMP_DIR=""
UV_CACHE_DIR=""
REPO_FULL_NAME=""

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/ci/build_and_push_uv_cache.sh [options]

Description:
  Builds a full prek uv cache locally and uploads it as a GitHub release asset.
  This script is intended to run on a machine/container compatible with the CI
  base image so cached native wheels can be reused in CI.

Options:
  --base-image <image>     CI base image metadata for fingerprint computation
  --python-mm <mm>         Python major.minor used in CI (default: 3.11)
  --build-jobs <n|auto>    Parallel native-build jobs while prebuilding cache (default: auto)
  --release-tag <tag>      GitHub release tag used to store cache assets (default: prek-uv-cache)
  --asset-prefix <prefix>  Asset prefix for cache archive names (default: prek-uv-cache)
  --keep <n>               Number of immutable cache assets to retain (default: 4)
  --part-size-mb <n>       Max size per uploaded cache part asset in MiB (default: 1900)
  --archive-path <path>    Optional output path for the generated cache archive
  --skip-build             Skip cache build and upload an existing --archive-path file
  --skip-prune             Skip old-asset pruning
  -h, --help               Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
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
    --release-tag)
      UV_CACHE_RELEASE_TAG="$2"
      shift 2
      ;;
    --asset-prefix)
      UV_CACHE_ASSET_PREFIX="$2"
      shift 2
      ;;
    --keep)
      KEEP_COUNT="$2"
      shift 2
      ;;
    --part-size-mb)
      PART_SIZE_MB="$2"
      shift 2
      ;;
    --archive-path)
      ARCHIVE_PATH="$2"
      shift 2
      ;;
    --skip-build)
      SKIP_BUILD=1
      shift
      ;;
    --skip-prune)
      SKIP_PRUNE=1
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

log() {
  printf '[ci-cache] %s\n' "$*"
}

fail() {
  printf '[ci-cache] ERROR: %s\n' "$*" >&2
  exit 1
}

require_cmd() {
  local cmd="$1"
  command -v "${cmd}" >/dev/null 2>&1 || fail "Required command not found: ${cmd}"
}

compute_fingerprint() {
  python3 "${REPO_ROOT}/scripts/ci/compute_uv_fingerprint.py" \
    --pyproject "${REPO_ROOT}/pyproject.toml" \
    --uv-lock "${REPO_ROOT}/uv.lock" \
    --base-image "${BASE_IMAGE}" \
    --python-mm "${PYTHON_MM}"
}

resolve_build_jobs() {
  local requested="$1"
  if [[ "${requested}" != "auto" ]]; then
    if ! [[ "${requested}" =~ ^[1-9][0-9]*$ ]]; then
      fail "build_jobs must be 'auto' or a positive integer."
    fi
    printf '%s\n' "${requested}"
    return
  fi

  local cpu_count
  cpu_count="$(nproc 2>/dev/null || echo 1)"
  local mem_kib
  mem_kib="$(awk '/MemTotal/ {print $2}' /proc/meminfo 2>/dev/null || echo 0)"
  local mem_gib="$((mem_kib / 1024 / 1024))"
  local mem_limited_jobs=1
  if ((mem_gib > 0)); then
    # Native CUDA extension builds are memory-intensive; keep ~3 GiB/job headroom.
    mem_limited_jobs="$((mem_gib / 3))"
    if ((mem_limited_jobs < 1)); then
      mem_limited_jobs=1
    fi
  fi
  if ((mem_limited_jobs > cpu_count)); then
    mem_limited_jobs="${cpu_count}"
  fi
  printf '%s\n' "${mem_limited_jobs}"
}

resolve_repo_full_name() {
  gh repo view --json nameWithOwner --jq '.nameWithOwner'
}

ensure_release_exists() {
  local repo="$1"
  if gh release view "${UV_CACHE_RELEASE_TAG}" --repo "${repo}" >/dev/null 2>&1; then
    return
  fi
  log "Creating release ${UV_CACHE_RELEASE_TAG} in ${repo}."
  gh release create "${UV_CACHE_RELEASE_TAG}" \
    --repo "${repo}" \
    --title "Prek CI uv cache" \
    --notes "Managed cache assets for prek CI dependency bootstrap."
}

build_cache_archive() {
  local archive_path="$1"
  local jobs="$2"

  TMP_DIR="$(mktemp -d)"
  UV_CACHE_DIR="${TMP_DIR}/uv-cache"
  mkdir -p "${UV_CACHE_DIR}"

  cp "${REPO_ROOT}/pyproject.toml" "${TMP_DIR}/pyproject.toml"
  cp "${REPO_ROOT}/uv.lock" "${TMP_DIR}/uv.lock"

  pushd "${TMP_DIR}" >/dev/null
  export UV_CACHE_DIR
  export UV_LINK_MODE=copy
  export UV_CONCURRENT_BUILDS="${jobs}"
  export CMAKE_BUILD_PARALLEL_LEVEL="${jobs}"
  export MAX_JOBS="${jobs}"
  export NINJAFLAGS="-j${jobs}"
  export TORCH_CUDA_ARCH_LIST=8.0

  local cudnn_path="${TMP_DIR}/.venv/lib/python${PYTHON_MM}/site-packages/nvidia/cudnn"
  export CUDNN_PATH="${cudnn_path}"
  export CUDNN_HOME="${cudnn_path}"
  export CUDNN_INCLUDE_PATH="${cudnn_path}/include"
  export CUDNN_LIBRARY_PATH="${cudnn_path}/lib"
  export CPLUS_INCLUDE_PATH="${CUDNN_INCLUDE_PATH}${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}}"
  export LIBRARY_PATH="${CUDNN_LIBRARY_PATH}${LIBRARY_PATH:+:${LIBRARY_PATH}}"
  export LD_LIBRARY_PATH="${CUDNN_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

  log "Building full uv cache with ${jobs} parallel jobs."
  uv sync --frozen --all-extras --group dev --no-install-project --python "${PYTHON_MM}"
  rm -rf .venv

  log "Packing uv cache archive to ${archive_path}."
  tar -C "${UV_CACHE_DIR}" -cf - . | zstd -6 -T0 -o "${archive_path}"
  popd >/dev/null

  rm -rf "${TMP_DIR}"
  TMP_DIR=""
  UV_CACHE_DIR=""
}

delete_assets_for_fingerprint() {
  local repo="$1"
  local fingerprint="$2"
  local release_json
  release_json="$(gh api "repos/${repo}/releases/tags/${UV_CACHE_RELEASE_TAG}")"
  local delete_ids
  delete_ids="$(RELEASE_JSON="${release_json}" PREFIX="${UV_CACHE_ASSET_PREFIX}" FINGERPRINT="${fingerprint}" python3 - <<'PY'
import json
import os
import re

payload = json.loads(os.environ["RELEASE_JSON"])
prefix = os.environ["PREFIX"]
fingerprint = os.environ["FINGERPRINT"]
pattern = re.compile(
    rf"^{re.escape(prefix)}-{re.escape(fingerprint)}\.tar\.zst(?:\.part-\d{{3}})?$"
)

for asset in payload.get("assets", []):
    name = asset.get("name", "")
    if pattern.match(name):
        print(asset["id"])
PY
)"

  if [[ -z "${delete_ids}" ]]; then
    return
  fi

  while IFS= read -r asset_id; do
    [[ -n "${asset_id}" ]] || continue
    log "Deleting stale cache asset id ${asset_id} for fingerprint ${fingerprint}."
    gh api --method DELETE "repos/${repo}/releases/assets/${asset_id}" >/dev/null
  done <<<"${delete_ids}"
}

upload_cache_assets() {
  local repo="$1"
  local archive_path="$2"
  local fingerprint="$3"

  [[ "${PART_SIZE_MB}" =~ ^[1-9][0-9]*$ ]] || fail "--part-size-mb must be a positive integer."
  if ((PART_SIZE_MB > 1900)); then
    fail "--part-size-mb must be <= 1900 to stay within GitHub release asset limits."
  fi

  delete_assets_for_fingerprint "${repo}" "${fingerprint}"

  local parts_dir
  parts_dir="$(mktemp -d)"
  local split_prefix="${parts_dir}/${UV_CACHE_ASSET_PREFIX}-${fingerprint}.tar.zst.part-"
  split --numeric-suffixes=0 --suffix-length=3 --bytes="${PART_SIZE_MB}m" "${archive_path}" "${split_prefix}"

  shopt -s nullglob
  local part_count=0
  local chunk
  for chunk in "${parts_dir}"/"${UV_CACHE_ASSET_PREFIX}-${fingerprint}.tar.zst.part-"*; do
    local part_asset="${chunk##*/}"
    log "Uploading cache part ${part_asset}."
    gh release upload "${UV_CACHE_RELEASE_TAG}" \
      --repo "${repo}" \
      "${chunk}" \
      --clobber
    part_count=$((part_count + 1))
  done
  shopt -u nullglob

  if ((part_count == 0)); then
    rm -rf "${parts_dir}"
    fail "No cache parts produced from archive ${archive_path}."
  fi
  rm -rf "${parts_dir}"
  printf '%s\n' "${part_count}"
}

prune_old_assets() {
  local repo="$1"
  local keep_count="$2"

  [[ "${keep_count}" =~ ^[1-9][0-9]*$ ]] || fail "--keep must be a positive integer."

  local release_json
  release_json="$(gh api "repos/${repo}/releases/tags/${UV_CACHE_RELEASE_TAG}")"
  local delete_ids
  delete_ids="$(RELEASE_JSON="${release_json}" PREFIX="${UV_CACHE_ASSET_PREFIX}" KEEP_COUNT="${keep_count}" python3 - <<'PY'
import json
import os
import re

payload = json.loads(os.environ["RELEASE_JSON"])
prefix = os.environ["PREFIX"]
keep_count = int(os.environ["KEEP_COUNT"])

assets = payload.get("assets", [])
by_fingerprint = {}
pattern_part = re.compile(rf"^{re.escape(prefix)}-([0-9a-f]+)\.tar\.zst\.part-\d{{3}}$")
pattern_single = re.compile(rf"^{re.escape(prefix)}-([0-9a-f]+)\.tar\.zst$")
for asset in assets:
    name = asset.get("name", "")
    match = pattern_part.match(name) or pattern_single.match(name)
    if not match:
        continue
    fingerprint = match.group(1)
    group = by_fingerprint.setdefault(fingerprint, {"latest": "", "ids": []})
    group["ids"].append(asset["id"])
    created_at = asset.get("created_at", "")
    if created_at > group["latest"]:
        group["latest"] = created_at

fingerprints = sorted(
    by_fingerprint.items(),
    key=lambda item: item[1]["latest"],
    reverse=True,
)
for _, group in fingerprints[keep_count:]:
    for asset_id in group["ids"]:
        print(asset_id)
PY
)"

  if [[ -z "${delete_ids}" ]]; then
    log "No old cache assets to prune."
    return
  fi

  while IFS= read -r asset_id; do
    [[ -n "${asset_id}" ]] || continue
    log "Deleting old cache asset id ${asset_id}."
    gh api --method DELETE "repos/${repo}/releases/assets/${asset_id}" >/dev/null
  done <<<"${delete_ids}"
}

main() {
  require_cmd gh
  require_cmd python3
  require_cmd uv
  require_cmd tar
  require_cmd zstd
  require_cmd split

  gh auth status >/dev/null 2>&1 || fail "GitHub auth not configured. Run: gh auth login"
  REPO_FULL_NAME="$(resolve_repo_full_name)"
  [[ -n "${REPO_FULL_NAME}" ]] || fail "Could not resolve gh repo."

  local fingerprint
  fingerprint="$(compute_fingerprint)"
  local jobs
  jobs="$(resolve_build_jobs "${BUILD_JOBS}")"
  local part_asset_prefix="${UV_CACHE_ASSET_PREFIX}-${fingerprint}.tar.zst.part-"
  local uploaded_part_count=""

  if [[ -z "${ARCHIVE_PATH}" ]]; then
    ARCHIVE_PATH="/tmp/${UV_CACHE_ASSET_PREFIX}-${fingerprint}.tar.zst"
  fi

  ensure_release_exists "${REPO_FULL_NAME}"

  if [[ "${SKIP_BUILD}" -eq 1 ]]; then
    [[ -f "${ARCHIVE_PATH}" ]] || fail "Archive not found for --skip-build: ${ARCHIVE_PATH}"
    log "Skipping build; using existing archive ${ARCHIVE_PATH}."
  else
    build_cache_archive "${ARCHIVE_PATH}" "${jobs}"
  fi

  uploaded_part_count="$(upload_cache_assets "${REPO_FULL_NAME}" "${ARCHIVE_PATH}" "${fingerprint}" | tail -n 1)"

  if [[ "${SKIP_PRUNE}" -eq 0 ]]; then
    prune_old_assets "${REPO_FULL_NAME}" "${KEEP_COUNT}"
  else
    log "Skipping asset pruning."
  fi

  cat <<MSG
[ci-cache] Uploaded prek uv cache asset.
[ci-cache] Repo:       ${REPO_FULL_NAME}
[ci-cache] Release:    ${UV_CACHE_RELEASE_TAG}
[ci-cache] Fingerprint:${fingerprint}
[ci-cache] Part prefix: ${part_asset_prefix}
[ci-cache] Parts:       ${uploaded_part_count}
[ci-cache] Archive:     ${ARCHIVE_PATH}
[ci-cache] Build jobs: ${jobs}
MSG
}

cleanup() {
  if [[ -n "${TMP_DIR}" && -d "${TMP_DIR}" ]]; then
    rm -rf "${TMP_DIR}"
  fi
}

trap cleanup EXIT

main
