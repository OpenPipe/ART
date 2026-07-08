#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: build-gpu-image.sh [options]

Options:
  --cluster-name NAME    Temporary BuildKit pod name to use
  --image-repo REPO      Image repository to publish
  --infra INFRA          Kubernetes-backed SkyPilot infra (default: k8s/cks-wb3)
  --no-cache             Disable registry-backed BuildKit cache
  --no-prewarm-modal     Skip prebuilding the pushed image in Modal
  --no-prewarm-nodes     Skip pre-pulling the pushed image on GPU nodes
  --prewarm-infra INFRA  Kubernetes-backed infra to prewarm; repeatable
  --pull-image-repo REPO Image repository for cluster pulls/prewarm
  --prewarm-modal        Require prebuilding the pushed image in Modal
  --prewarm-timeout DUR  Timeout for the prewarm DaemonSet rollout (default: 30m)
  --tag TAG              Image tag to publish (default: latest)
  --help                 Show this help
EOF
}

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

cluster_name=""
infra="${SKY_INFRA:-k8s/cks-wb3}"
image_repo="${ART_IMAGE_REPO:-}"
pull_image_repo="${ART_PULL_IMAGE_REPO:-}"
image_tag="${IMAGE_TAG:-latest}"
docker_config_path="${DOCKER_CONFIG_PATH:-${HOME}/.docker/config.json}"
buildkit_image="${BUILDKIT_IMAGE:-moby/buildkit:v0.29.0-rootless}"
buildkit_namespace="${KUBECTL_NAMESPACE:-default}"
buildkit_wait_timeout="${BUILDKIT_WAIT_TIMEOUT:-300s}"
no_cache="${NO_CACHE:-false}"
prewarm_modal="${PREWARM_MODAL:-auto}"
prewarm_nodes="${PREWARM_NODES:-true}"
prewarm_infras=()
if [[ -n "${PREWARM_INFRAS:-}" ]]; then
  while IFS= read -r prewarm_infra; do
    [[ -n "${prewarm_infra}" ]] && prewarm_infras+=("${prewarm_infra}")
  done < <(printf '%s\n' "${PREWARM_INFRAS}" | awk 'BEGIN { RS = "[[:space:],]+" } NF { print }')
fi
prewarm_namespace="${PREWARM_NAMESPACE:-default}"
prewarm_name="${PREWARM_NAME:-art-gpu-image-prewarm}"
prewarm_image_pull_secret="${PREWARM_IMAGE_PULL_SECRET:-art-gpu-registry-auth}"
prewarm_node_selector="${PREWARM_NODE_SELECTOR:-node.coreweave.cloud/class=gpu}"
prewarm_timeout="${PREWARM_TIMEOUT:-30m}"
prewarm_node_timeout="${PREWARM_NODE_TIMEOUT:-10m}"
prewarm_delete_timeout="${PREWARM_DELETE_TIMEOUT:-60s}"
prewarm_node_retries="${PREWARM_NODE_RETRIES:-12}"
prewarm_node_parallelism="${PREWARM_NODE_PARALLELISM:-3}"
prewarm_tag_convergence_attempts="${PREWARM_TAG_CONVERGENCE_ATTEMPTS:-120}"
prewarm_tag_convergence_interval="${PREWARM_TAG_CONVERGENCE_INTERVAL:-15}"
prewarm_clear_mutable_tag="${PREWARM_CLEAR_MUTABLE_TAG:-true}"
prewarm_crictl_image="${PREWARM_CRICTL_IMAGE:-alpine:3.20}"
prewarm_crictl_version="${PREWARM_CRICTL_VERSION:-v1.36.0}"
prewarm_containerd_socket="${PREWARM_CONTAINERD_SOCKET:-/run/containerd/containerd.sock}"
prewarm_containerd_mount="${prewarm_containerd_socket%/*}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cluster-name)
      cluster_name="$2"
      shift 2
      ;;
    --image-repo)
      image_repo="$2"
      shift 2
      ;;
    --infra)
      infra="$2"
      shift 2
      ;;
    --no-cache)
      no_cache=true
      shift
      ;;
    --no-prewarm-modal)
      prewarm_modal=false
      shift
      ;;
    --no-prewarm-nodes)
      prewarm_nodes=false
      shift
      ;;
    --prewarm-infra)
      prewarm_infras+=("$2")
      shift 2
      ;;
    --pull-image-repo)
      pull_image_repo="$2"
      shift 2
      ;;
    --prewarm-modal)
      prewarm_modal=true
      shift
      ;;
    --prewarm-timeout)
      prewarm_timeout="$2"
      shift 2
      ;;
    --tag)
      image_tag="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

case "${prewarm_modal}" in
  auto|true|false) ;;
  *)
    echo "PREWARM_MODAL must be one of: auto, true, false" >&2
    exit 1
    ;;
esac

kube_context_from_infra() {
  local infra_value="$1"
  case "${infra_value}" in
    k8s/*)
      printf '%s\n' "${infra_value#k8s/}"
      ;;
    kubernetes/*)
      printf '%s\n' "${infra_value#kubernetes/}"
      ;;
    *)
      echo "Unsupported infra '${infra_value}'. Use k8s/<kubectl-context>." >&2
      return 1
      ;;
  esac
}

kube_context="$(kube_context_from_infra "${infra}")"
build_kubectl_cmd=(kubectl --context "${kube_context}")
kubectl_cmd=("${build_kubectl_cmd[@]}")
if [[ "${#prewarm_infras[@]}" == "0" ]]; then
  prewarm_infras=("${infra}")
fi
prewarm_contexts=()
for prewarm_infra in "${prewarm_infras[@]}"; do
  prewarm_contexts+=("$(kube_context_from_infra "${prewarm_infra}")")
done
if [[ "${prewarm_node_selector}" != *=* ]]; then
  echo "PREWARM_NODE_SELECTOR must be a key=value selector, got: ${prewarm_node_selector}" >&2
  exit 1
fi
prewarm_node_selector_key="${prewarm_node_selector%%=*}"
prewarm_node_selector_value="${prewarm_node_selector#*=}"

art_sha="$(git -C "${repo_root}" rev-parse HEAD)"
art_short_sha="$(git -C "${repo_root}" rev-parse --short=12 HEAD)"
timestamp="$(date +%m%d-%H%M%S)"

if [[ -z "${cluster_name}" ]]; then
  cluster_name="art-gpu-build-${timestamp}"
fi

dockerhub_user=""
if [[ -f "${docker_config_path}" ]]; then
  export DOCKER_CONFIG_PATH="${docker_config_path}"
  dockerhub_user="$(
    uv run --no-project python - <<'PY'
import base64
import json
import os

path = os.environ["DOCKER_CONFIG_PATH"]
data = json.load(open(path))
auths = data.get("auths", {})
for key in (
    "https://index.docker.io/v1/",
    "https://index.docker.io/v1/access-token",
    "https://index.docker.io/v1/refresh-token",
):
    entry = auths.get(key)
    if entry and "auth" in entry:
        print(base64.b64decode(entry["auth"]).decode().split(":", 1)[0])
        break
PY
  )"
fi

if [[ -z "${image_repo}" ]]; then
  if [[ -n "${dockerhub_user}" ]]; then
    image_repo="docker.io/${dockerhub_user}/art-gpu"
  else
    image_repo="ghcr.io/openpipe/art-gpu"
  fi
fi
if [[ -z "${pull_image_repo}" ]]; then
  pull_image_repo="${image_repo}"
fi

registry_host="${image_repo%%/*}"
if [[ -z "${registry_host}" || "${registry_host}" == "${image_repo}" ||
  ( "${registry_host}" != *.* && "${registry_host}" != *:* && "${registry_host}" != "localhost" ) ]]; then
  registry_host="docker.io"
fi
cache_ref="${BUILDKIT_CACHE_REF:-${image_repo}:buildcache}"
cache_opts=""
if [[ "${no_cache}" != "true" ]]; then
  cache_opts="--import-cache type=registry,ref=${cache_ref} --export-cache type=registry,ref=${cache_ref},mode=max"
fi

if [[ -n "${REGISTRY_AUTH_JSON_B64:-}" ]]; then
  registry_auth_json_b64="${REGISTRY_AUTH_JSON_B64}"
elif [[ "${registry_host}" == "docker.io" && -f "${docker_config_path}" ]]; then
  dockerhub_auth_json="$(
    DOCKER_CONFIG_PATH="${docker_config_path}" uv run --no-project python - <<'PY'
import base64
import json
import os
import urllib.request

path = os.environ["DOCKER_CONFIG_PATH"]
data = json.load(open(path))
auths = data.get("auths", {})
basic_auth = None
for key in (
    "https://index.docker.io/v1/",
    "docker.io",
    "index.docker.io/v1/",
):
    entry = auths.get(key)
    if entry and "auth" in entry:
        basic_auth = entry["auth"]
        break

if basic_auth is None:
    raise SystemExit(f"Missing Docker Hub auth entry in {path}")

username, password = base64.b64decode(basic_auth).decode().split(":", 1)
login_req = urllib.request.Request(
    "https://hub.docker.com/v2/users/login/",
    data=json.dumps({"username": username, "password": password}).encode(),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(login_req, timeout=30) as resp:
    login_payload = json.load(resp)

access_auth = base64.b64encode(f"{username}:{login_payload['token']}".encode()).decode()
refresh_auth = base64.b64encode(f"{username}:{login_payload['refresh_token']}".encode()).decode()

print(
    json.dumps(
        {
            "auths": {
                "https://index.docker.io/v1/": {"auth": basic_auth},
                "docker.io": {"auth": basic_auth},
                "https://registry-1.docker.io/v2/": {"auth": basic_auth},
                "registry-1.docker.io": {"auth": basic_auth},
                "https://index.docker.io/v1/access-token": {"auth": access_auth},
                "https://index.docker.io/v1/refresh-token": {"auth": refresh_auth},
            }
        },
        separators=(",", ":"),
    )
)
PY
  )"
  registry_auth_json_b64="$(printf '%s' "${dockerhub_auth_json}" | base64 | tr -d '\n')"
else
  ghcr_username="${GHCR_USERNAME:-$(gh api user --jq .login)}"
  ghcr_token="${GHCR_TOKEN:-$(gh auth token)}"
  ghcr_auth="$(printf '%s' "${ghcr_username}:${ghcr_token}" | base64 | tr -d '\n')"
  registry_auth_json_b64="$(
    printf '{"auths":{"ghcr.io":{"auth":"%s"}}}' "${ghcr_auth}" | base64 | tr -d '\n'
  )"
fi

context_dir="$(mktemp -d "${TMPDIR:-/tmp}/art-gpu-build-context.XXXXXX")"
buildkit_manifest_path="$(mktemp "${TMPDIR:-/tmp}/art-gpu-buildkit.XXXXXX")"
registry_auth_json_path="$(mktemp "${TMPDIR:-/tmp}/art-gpu-auth.XXXXXX")"
build_command_path="$(mktemp "${TMPDIR:-/tmp}/art-gpu-build-command.XXXXXX")"
build_log_snapshot_path="$(mktemp "${TMPDIR:-/tmp}/art-gpu-build-log.XXXXXX")"
build_log_offset_path="$(mktemp "${TMPDIR:-/tmp}/art-gpu-build-log-offset.XXXXXX")"
cleanup() {
  rm -rf "${context_dir}"
  rm -f "${buildkit_manifest_path}" "${registry_auth_json_path}" \
    "${build_command_path}" "${build_log_snapshot_path}" "${build_log_offset_path}"
  "${build_kubectl_cmd[@]}" delete pod -n "${buildkit_namespace}" "${cluster_name}" \
    --ignore-not-found --wait=true >/dev/null 2>&1 || true
}
trap cleanup EXIT
printf '0' > "${build_log_offset_path}"

mkdir -p "${context_dir}/docker" "${context_dir}/vllm_runtime"
cp "${repo_root}/pyproject.toml" "${context_dir}/pyproject.toml"
cp "${repo_root}/uv.lock" "${context_dir}/uv.lock"
cp "${repo_root}/vllm_runtime/pyproject.toml" "${context_dir}/vllm_runtime/pyproject.toml"
cp "${repo_root}/vllm_runtime/uv.lock" "${context_dir}/vllm_runtime/uv.lock"
cp "${repo_root}/.dockerignore" "${context_dir}/.dockerignore"
cp "${repo_root}/docker/art-gpu.Dockerfile" "${context_dir}/docker/art-gpu.Dockerfile"
printf '%s' "${registry_auth_json_b64}" | base64 -d > "${registry_auth_json_path}"

echo "Launching temporary BuildKit pod ${cluster_name} on ${infra}"
echo "Publishing ${image_repo}:${image_tag}"
echo "Cluster pull image ${pull_image_repo}:${image_tag}"
if [[ "${no_cache}" == "true" ]]; then
  echo "Registry cache disabled"
else
  echo "Using registry cache ${cache_ref}"
fi
echo "Using ART_SHA=${art_sha}"

cat > "${buildkit_manifest_path}" <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: ${cluster_name}
  annotations:
    container.apparmor.security.beta.kubernetes.io/buildkitd: unconfined
spec:
  restartPolicy: Never
  containers:
    - name: buildkitd
      image: ${buildkit_image}
      args:
        - --oci-worker-no-process-sandbox
      readinessProbe:
        exec:
          command: ["buildctl", "debug", "workers"]
        initialDelaySeconds: 5
        periodSeconds: 5
      securityContext:
        seccompProfile:
          type: Unconfined
        runAsUser: 1000
        runAsGroup: 1000
      volumeMounts:
        - mountPath: /home/user/.local/share/buildkit
          name: buildkitd
  volumes:
    - name: buildkitd
      emptyDir: {}
EOF

"${kubectl_cmd[@]}" delete pod -n "${buildkit_namespace}" "${cluster_name}" \
  --ignore-not-found --wait=true >/dev/null 2>&1 || true
"${kubectl_cmd[@]}" apply -n "${buildkit_namespace}" -f "${buildkit_manifest_path}"
"${kubectl_cmd[@]}" wait -n "${buildkit_namespace}" \
  --for=condition=Ready "pod/${cluster_name}" \
  --timeout="${buildkit_wait_timeout}"
"${kubectl_cmd[@]}" exec -n "${buildkit_namespace}" "${cluster_name}" -- sh -lc \
  'mkdir -p /home/user/.docker /tmp/build-context'
"${kubectl_cmd[@]}" cp "${registry_auth_json_path}" \
  "${buildkit_namespace}/${cluster_name}:/home/user/.docker/config.json"
"${kubectl_cmd[@]}" cp "${context_dir}/." \
  "${buildkit_namespace}/${cluster_name}:/tmp/build-context"

cat > "${build_command_path}" <<EOF
#!/bin/sh
set -eu
buildctl build \
    --progress=plain \
    ${cache_opts} \
    --frontend dockerfile.v0 \
    --local context=/tmp/build-context \
    --local dockerfile=/tmp/build-context/docker \
    --opt filename=art-gpu.Dockerfile \
    --opt build-arg:ART_SHA=${art_sha} \
    --output type=image,name=${image_repo}:${image_tag},push=true
EOF

sync_build_log() {
  if "${kubectl_cmd[@]}" cp \
    "${buildkit_namespace}/${cluster_name}:/tmp/art-build.log" \
    "${build_log_snapshot_path}" >/dev/null 2>&1; then
    uv run --no-project python - "${build_log_snapshot_path}" "${build_log_offset_path}" <<'PY'
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
offset_path = Path(sys.argv[2])
offset = int(offset_path.read_text() or "0")
data = log_path.read_bytes()
if offset < len(data):
    sys.stdout.buffer.write(data[offset:])
    sys.stdout.flush()
offset_path.write_text(str(len(data)))
PY
  fi
}

"${kubectl_cmd[@]}" cp "${build_command_path}" \
  "${buildkit_namespace}/${cluster_name}:/tmp/art-build.sh"
"${kubectl_cmd[@]}" exec -n "${buildkit_namespace}" "${cluster_name}" -- sh -lc '
  chmod +x /tmp/art-build.sh
  rm -f /tmp/art-build.log /tmp/art-build.exit
  nohup sh -c '"'"'/tmp/art-build.sh >/tmp/art-build.log 2>&1; printf "%s\n" "$?" >/tmp/art-build.exit'"'"' >/tmp/art-build.nohup 2>&1 &
'

while true; do
  sync_build_log
  build_exit_code="$(
    "${kubectl_cmd[@]}" exec -n "${buildkit_namespace}" "${cluster_name}" -- sh -lc \
      'if [ -f /tmp/art-build.exit ]; then sed -n 1p /tmp/art-build.exit; fi' 2>/dev/null || true
  )"
  if [[ -n "${build_exit_code}" ]]; then
    sync_build_log
    if [[ "${build_exit_code}" != "0" ]]; then
      exit "${build_exit_code}"
    fi
    break
  fi
  sleep 10
done

echo
echo "Image ready for testing:"
echo "  ${image_repo}:${image_tag}"
if [[ "${pull_image_repo}" != "${image_repo}" ]]; then
  echo "Cluster pull image:"
  echo "  ${pull_image_repo}:${image_tag}"
fi
image_digest="$(
  uv run --no-project python - "${build_log_snapshot_path}" "${image_repo}:${image_tag}" <<'PY'
import re
import sys
from pathlib import Path

log = Path(sys.argv[1]).read_text(errors="replace")
image = re.escape(sys.argv[2])
matches = re.findall(rf"pushing manifest for {image}@(sha256:[0-9a-f]+)", log)
if matches:
    print(matches[-1])
PY
)"
prewarm_tag_image="${pull_image_repo}:${image_tag}"
prewarm_image="${prewarm_tag_image}"
prewarm_refresh_tag_image=""
if [[ -n "${image_digest}" ]]; then
  if [[ "${pull_image_repo}" == "${image_repo}" ]]; then
    prewarm_image="${pull_image_repo}@${image_digest}"
    prewarm_refresh_tag_image="${prewarm_tag_image}"
  else
    echo "Prewarm pull repo differs from pushed image repo; using mutable tag for pull-through freshness:"
    echo "  Pushed image digest: ${image_repo}@${image_digest}"
    echo "  Prewarm image: ${prewarm_image}"
  fi
fi
prewarm_display="${prewarm_image}"
if [[ -n "${prewarm_refresh_tag_image}" ]]; then
  prewarm_display="${prewarm_image} and refreshing ${prewarm_refresh_tag_image}"
fi
prewarm_clear_tag_images=()
prewarm_short_tag_image=""
if [[ -n "${prewarm_refresh_tag_image}" ]]; then
  prewarm_clear_tag_images+=("${prewarm_refresh_tag_image}")
  case "${prewarm_refresh_tag_image}" in
    docker.io/*)
      prewarm_short_tag_image="${prewarm_refresh_tag_image#docker.io/}"
      prewarm_clear_tag_images+=(
        "${prewarm_short_tag_image}"
        "images.coreweave.com/cluster-images/${prewarm_refresh_tag_image#docker.io/}"
      )
      ;;
    registry-1.docker.io/*)
      prewarm_short_tag_image="${prewarm_refresh_tag_image#registry-1.docker.io/}"
      prewarm_clear_tag_images+=(
        "${prewarm_short_tag_image}"
        "images.coreweave.com/cluster-images/${prewarm_refresh_tag_image#registry-1.docker.io/}"
      )
      ;;
  esac
fi

modal_auth_available=false
if [[ "${prewarm_modal}" != "false" ]]; then
  if uv run --with 'modal>=1.5.0' python - <<'PY' >/dev/null 2>&1; then
import modal

modal.Workspace.from_context().hydrate()
PY
    modal_auth_available=true
  fi
fi

if [[ "${prewarm_modal}" == "true" || "${modal_auth_available}" == "true" ]]; then
  echo "Prewarming ${image_repo}:${image_tag} in Modal image cache"
  MODAL_FORCE_BUILD=1 uv run --with 'modal>=1.5.0' python - "${image_repo}:${image_tag}" <<'PY'
import sys

import modal

image = (
    modal.Image.from_registry(sys.argv[1], add_python="3.12")
    .apt_install("openssh-server", "sudo", "rsync", "curl", "procps", "patch", "lsof")
)
app = modal.App.lookup("skypilot-modal", create_if_missing=True)
with modal.enable_output():
    image.build(app)
PY
elif [[ "${prewarm_modal}" == "auto" ]]; then
  echo "Skipping Modal image prewarm: Modal auth unavailable"
else
  echo "Skipping Modal image prewarm"
fi

dump_prewarm_diagnostics() {
  echo "::group::Prewarm diagnostics"
  "${kubectl_cmd[@]}" get daemonset -n "${prewarm_namespace}" "${prewarm_name}" -o wide || true
  "${kubectl_cmd[@]}" get pods -n "${prewarm_namespace}" -l "app=${prewarm_name}" -o wide || true
  "${kubectl_cmd[@]}" get pods -n "${prewarm_namespace}" -l "art.openpipe/prewarm-name=${prewarm_name}" -o wide || true
  "${kubectl_cmd[@]}" describe daemonset -n "${prewarm_namespace}" "${prewarm_name}" || true
  first_prewarm_pod="$(
    "${kubectl_cmd[@]}" get pods -n "${prewarm_namespace}" -l "app=${prewarm_name}" \
      -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true
  )"
  if [[ -n "${first_prewarm_pod}" ]]; then
    "${kubectl_cmd[@]}" describe pod -n "${prewarm_namespace}" "${first_prewarm_pod}" || true
  fi
  "${kubectl_cmd[@]}" get events -n "${prewarm_namespace}" --sort-by=.lastTimestamp | tail -n 80 || true
  echo "::endgroup::"
}

sanitize_k8s_name_part() {
  printf '%s' "$1" \
    | tr '[:upper:]' '[:lower:]' \
    | tr -c 'a-z0-9-' '-' \
    | sed -E 's/^-+//; s/-+$//; s/-+/-/g' \
    | cut -c1-35
}

render_prewarm_init_containers() {
  if [[ -n "${prewarm_refresh_tag_image}" && "${prewarm_clear_mutable_tag}" == "true" ]]; then
    cat <<EOF
    - name: clear-tag
      image: ${prewarm_crictl_image}
      imagePullPolicy: IfNotPresent
      securityContext:
        privileged: true
      command:
        - sh
        - -lc
        - |
          set -eu
          wget -qO /tmp/crictl.tar.gz https://github.com/kubernetes-sigs/cri-tools/releases/download/${prewarm_crictl_version}/crictl-${prewarm_crictl_version}-linux-amd64.tar.gz
          tar -xzf /tmp/crictl.tar.gz -C /usr/local/bin crictl
$(render_clear_tag_commands)
      volumeMounts:
        - name: containerd-socket
          mountPath: ${prewarm_containerd_mount}
      resources:
        requests:
          cpu: 10m
          memory: 32Mi
EOF
  fi
  cat <<EOF
    - name: prepull
      image: ${prewarm_image}
      imagePullPolicy: Always
      command: ["bash", "-lc", "true"]
      resources:
        requests:
          cpu: 10m
          memory: 16Mi
EOF
  if [[ -n "${prewarm_refresh_tag_image}" && "${prewarm_refresh_tag_image}" != "${prewarm_image}" ]]; then
    cat <<EOF
    - name: refresh-tag
      image: ${prewarm_refresh_tag_image}
      imagePullPolicy: Always
      command: ["bash", "-lc", "true"]
      resources:
        requests:
          cpu: 10m
          memory: 16Mi
EOF
  fi
  if [[ -n "${prewarm_short_tag_image}" ]]; then
    cat <<EOF
    - name: refresh-short-tag
      image: ${prewarm_short_tag_image}
      imagePullPolicy: Always
      command: ["bash", "-lc", "true"]
      resources:
        requests:
          cpu: 10m
          memory: 16Mi
EOF
  fi
}

render_prewarm_host_volumes() {
  if [[ -n "${prewarm_refresh_tag_image}" && "${prewarm_clear_mutable_tag}" == "true" ]]; then
    cat <<EOF
volumes:
  - name: containerd-socket
    hostPath:
      path: ${prewarm_containerd_mount}
      type: Directory
EOF
  fi
}

render_clear_tag_commands() {
  local image

  for image in "${prewarm_clear_tag_images[@]}"; do
    printf '          crictl --runtime-endpoint "unix://%s" rmi '\''%s'\'' || true\n' \
      "${prewarm_containerd_socket}" "${image}"
  done
}

delete_pods_best_effort() {
  if (( $# == 0 )); then
    return 0
  fi

  if "${kubectl_cmd[@]}" delete pod -n "${prewarm_namespace}" "$@" \
    --ignore-not-found --wait=true --timeout="${prewarm_delete_timeout}" >/dev/null 2>&1; then
    return 0
  fi

  "${kubectl_cmd[@]}" delete pod -n "${prewarm_namespace}" "$@" \
    --ignore-not-found --wait=false --grace-period=0 --force >/dev/null 2>&1 || true
}

delete_pods_without_wait() {
  if (( $# == 0 )); then
    return 0
  fi

  "${kubectl_cmd[@]}" delete pod -n "${prewarm_namespace}" "$@" \
    --ignore-not-found --wait=false >/dev/null 2>&1 || true
}

render_tag_check_init_containers() {
  local clear_mutable_tag="${1:-${prewarm_clear_mutable_tag}}"

  if [[ -n "${prewarm_refresh_tag_image}" && "${clear_mutable_tag}" == "true" ]]; then
    cat <<EOF
    - name: clear-tag
      image: ${prewarm_crictl_image}
      imagePullPolicy: IfNotPresent
      securityContext:
        privileged: true
      command:
        - sh
        - -lc
        - |
          set -eu
          wget -qO /tmp/crictl.tar.gz https://github.com/kubernetes-sigs/cri-tools/releases/download/${prewarm_crictl_version}/crictl-${prewarm_crictl_version}-linux-amd64.tar.gz
          tar -xzf /tmp/crictl.tar.gz -C /usr/local/bin crictl
$(render_clear_tag_commands)
      volumeMounts:
        - name: containerd-socket
          mountPath: ${prewarm_containerd_mount}
      resources:
        requests:
          cpu: 10m
          memory: 32Mi
EOF
  fi
  cat <<EOF
    - name: refresh-tag
      image: ${prewarm_refresh_tag_image}
      imagePullPolicy: Always
      command: ["bash", "-lc", "true"]
      resources:
        requests:
          cpu: 10m
          memory: 16Mi
EOF
  if [[ -n "${prewarm_short_tag_image}" ]]; then
    cat <<EOF
    - name: refresh-short-tag
      image: ${prewarm_short_tag_image}
      imagePullPolicy: Always
      command: ["bash", "-lc", "true"]
      resources:
        requests:
          cpu: 10m
          memory: 16Mi
EOF
  fi
}

pod_init_image_id() {
  local pod="$1"
  local container="$2"

  "${kubectl_cmd[@]}" get pod -n "${prewarm_namespace}" "${pod}" \
    -o jsonpath="{range .status.initContainerStatuses[?(@.name==\"${container}\")]}{.imageID}{end}" \
    2>/dev/null || true
}

verify_prewarm_refresh_tag_digest() {
  local pod="$1"
  local image_id
  local failed=0

  if [[ -z "${prewarm_refresh_tag_image}" || -z "${image_digest}" ]]; then
    return 0
  fi

  image_id="$(pod_init_image_id "${pod}" refresh-tag)"
  if [[ "${image_id}" == *"@${image_digest}" ]]; then
    echo "Mutable tag ${prewarm_refresh_tag_image} resolved to ${image_id} in pod ${pod}"
  else
    echo "Mutable tag ${prewarm_refresh_tag_image} resolved to ${image_id:-<missing>} in pod ${pod}; expected @${image_digest}" >&2
    failed=1
  fi

  if [[ -n "${prewarm_short_tag_image}" ]]; then
    image_id="$(pod_init_image_id "${pod}" refresh-short-tag)"
    if [[ "${image_id}" == *"@${image_digest}" ]]; then
      echo "Mutable tag ${prewarm_short_tag_image} resolved to ${image_id} in pod ${pod}"
    else
      echo "Mutable tag ${prewarm_short_tag_image} resolved to ${image_id:-<missing>} in pod ${pod}; expected @${image_digest}" >&2
      failed=1
    fi
  fi

  return "${failed}"
}

verify_steady_prewarm_daemonset() {
  local attempt
  local steady_pod
  local -a steady_pods
  local -a stale_pods

  for attempt in $(seq 1 "${prewarm_node_retries}"); do
    if ! "${kubectl_cmd[@]}" rollout status -n "${prewarm_namespace}" "daemonset/${prewarm_name}" --timeout="${prewarm_timeout}"; then
      return 1
    fi

    mapfile -t steady_pods < <(
      "${kubectl_cmd[@]}" get pods -n "${prewarm_namespace}" -l "app=${prewarm_name}" \
        -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' 2>/dev/null
    )
    stale_pods=()
    for steady_pod in "${steady_pods[@]}"; do
      if ! verify_prewarm_refresh_tag_digest "${steady_pod}"; then
        stale_pods+=("${steady_pod}")
      fi
    done

    if (( ${#stale_pods[@]} == 0 )); then
      return 0
    fi

    if (( attempt == prewarm_node_retries )); then
      echo "Steady-state ${prewarm_name} mutable tag verification failed after ${prewarm_node_retries} attempt(s)" >&2
      return 1
    fi

    echo "Steady-state ${prewarm_name} has ${#stale_pods[@]} pod(s) with stale ${prewarm_refresh_tag_image}; recreating them (attempt ${attempt}/${prewarm_node_retries})"
    delete_pods_best_effort "${stale_pods[@]}"
    sleep "$((attempt * 10))"
  done
}

wait_for_mutable_tag_convergence() {
  local node="$1"
  local pod_base="${prewarm_name}-tag-check"
  local pod
  local attempt
  local clear_mutable_tag
  local image_id

  if [[ -z "${prewarm_refresh_tag_image}" || -z "${image_digest}" ]]; then
    return 0
  fi
  if ! [[ "${prewarm_tag_convergence_attempts}" =~ ^[1-9][0-9]*$ ]]; then
    echo "PREWARM_TAG_CONVERGENCE_ATTEMPTS must be a positive integer, got: ${prewarm_tag_convergence_attempts}" >&2
    exit 1
  fi
  if ! [[ "${prewarm_tag_convergence_interval}" =~ ^[1-9][0-9]*$ ]]; then
    echo "PREWARM_TAG_CONVERGENCE_INTERVAL must be a positive integer number of seconds, got: ${prewarm_tag_convergence_interval}" >&2
    exit 1
  fi

  echo "Waiting for ${prewarm_refresh_tag_image} to resolve to ${image_digest} from Kubernetes pulls"
  for attempt in $(seq 1 "${prewarm_tag_convergence_attempts}"); do
    clear_mutable_tag=false
    if [[ "${attempt}" == "1" ]]; then
      clear_mutable_tag="${prewarm_clear_mutable_tag}"
    fi
    pod="${pod_base}-${attempt}"
    delete_pods_without_wait "${pod}"
    "${kubectl_cmd[@]}" apply -n "${prewarm_namespace}" -f - <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: ${pod}
  labels:
    app: ${prewarm_name}-oneshot
    art.openpipe/prewarm-name: ${prewarm_name}
    art.openpipe/prewarm-token: "${timestamp}-${art_short_sha}"
spec:
  restartPolicy: Never
  nodeName: ${node}
  imagePullSecrets:
    - name: ${prewarm_image_pull_secret}
  tolerations:
    - operator: Exists
  initContainers:
$(render_tag_check_init_containers "${clear_mutable_tag}")
  containers:
    - name: pause
      image: registry.k8s.io/pause:3.10
      resources:
        requests:
          cpu: 10m
          memory: 16Mi
$(render_prewarm_host_volumes | sed 's/^/  /')
EOF

    if "${kubectl_cmd[@]}" wait -n "${prewarm_namespace}" \
      --for=condition=Ready "pod/${pod}" \
      --timeout="${prewarm_node_timeout}" >/dev/null 2>&1; then
      if verify_prewarm_refresh_tag_digest "${pod}"; then
        echo "Mutable tags converged to ${image_digest}"
        delete_pods_without_wait "${pod}"
        return 0
      fi
      echo "Mutable tag check ${attempt}/${prewarm_tag_convergence_attempts} has not converged to ${image_digest}"
    else
      echo "Mutable tag check ${attempt}/${prewarm_tag_convergence_attempts} did not become ready"
      "${kubectl_cmd[@]}" describe pod -n "${prewarm_namespace}" "${pod}" || true
    fi

    delete_pods_without_wait "${pod}"
    sleep "${prewarm_tag_convergence_interval}"
  done

  echo "Timed out waiting for ${prewarm_refresh_tag_image} to resolve to ${image_digest}" >&2
  return 1
}

prewarm_single_node() {
  local node="$1"
  local node_slug
  local pod
  local attempt

  node_slug="$(sanitize_k8s_name_part "${node}")"
  if [[ -z "${node_slug}" ]]; then
    echo "Could not derive Kubernetes pod name for node ${node}" >&2
    return 1
  fi
  pod="${prewarm_name}-${node_slug}"

  for attempt in $(seq 1 "${prewarm_node_retries}"); do
    echo "Prewarming ${prewarm_display} on GPU node ${node} (attempt ${attempt}/${prewarm_node_retries})"
    delete_pods_best_effort "${pod}"
    "${kubectl_cmd[@]}" apply -n "${prewarm_namespace}" -f - <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: ${pod}
  labels:
    app: ${prewarm_name}-oneshot
    art.openpipe/prewarm-name: ${prewarm_name}
    art.openpipe/prewarm-token: "${timestamp}-${art_short_sha}"
spec:
  restartPolicy: Never
  nodeName: ${node}
  imagePullSecrets:
    - name: ${prewarm_image_pull_secret}
  tolerations:
    - operator: Exists
  initContainers:
$(render_prewarm_init_containers)
  containers:
    - name: pause
      image: registry.k8s.io/pause:3.10
      resources:
        requests:
          cpu: 10m
          memory: 16Mi
$(render_prewarm_host_volumes | sed 's/^/  /')
EOF
    if "${kubectl_cmd[@]}" wait -n "${prewarm_namespace}" \
      --for=condition=Ready "pod/${pod}" \
      --timeout="${prewarm_node_timeout}"; then
      if verify_prewarm_refresh_tag_digest "${pod}"; then
        delete_pods_best_effort "${pod}"
        return 0
      fi

      echo "Prewarm mutable tag verification failed on node ${node}; retrying after tag convergence delay"
      delete_pods_best_effort "${pod}"
      sleep "$((attempt * 10))"
      continue
    fi

    echo "Prewarm failed on node ${node}; pod diagnostics:"
    "${kubectl_cmd[@]}" describe pod -n "${prewarm_namespace}" "${pod}" || true
    delete_pods_best_effort "${pod}"
    sleep "$((attempt * 10))"
  done

  echo "Failed to prewarm ${prewarm_image} on node ${node}" >&2
  return 1
}

if [[ "${prewarm_nodes}" == "true" ]]; then
  for prewarm_context in "${prewarm_contexts[@]}"; do
    kubectl_cmd=(kubectl --context "${prewarm_context}")
    echo "Prewarming Kubernetes context ${prewarm_context}"
  if ! [[ "${prewarm_node_parallelism}" =~ ^[1-9][0-9]*$ ]]; then
    echo "PREWARM_NODE_PARALLELISM must be a positive integer, got: ${prewarm_node_parallelism}" >&2
    exit 1
  fi
  mapfile -t gpu_nodes < <(
    "${kubectl_cmd[@]}" get nodes -l "${prewarm_node_selector}" \
      -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' 2>/dev/null
  )
  gpu_node_count="${#gpu_nodes[@]}"
  if [[ "${gpu_node_count}" == "0" ]]; then
    echo "Skipping GPU node prewarm: no nodes match ${prewarm_node_selector}"
  else
    echo "Prewarming ${prewarm_display} on ${gpu_node_count} GPU node(s)"
    "${kubectl_cmd[@]}" create secret generic "${prewarm_image_pull_secret}" \
      -n "${prewarm_namespace}" \
      --from-file=.dockerconfigjson="${registry_auth_json_path}" \
      --type=kubernetes.io/dockerconfigjson \
      --dry-run=client -o yaml \
      | "${kubectl_cmd[@]}" apply -n "${prewarm_namespace}" -f -

    echo "Stopping existing ${prewarm_name} DaemonSet before batched node prewarm"
    "${kubectl_cmd[@]}" delete daemonset -n "${prewarm_namespace}" "${prewarm_name}" \
      --ignore-not-found --wait=true >/dev/null 2>&1 || true

    if ! wait_for_mutable_tag_convergence "${gpu_nodes[0]}"; then
      dump_prewarm_diagnostics
      exit 1
    fi

    prewarm_failures=0
    prewarm_pids=()
    for gpu_node in "${gpu_nodes[@]}"; do
      prewarm_single_node "${gpu_node}" &
      prewarm_pids+=("$!")

      if (( ${#prewarm_pids[@]} >= prewarm_node_parallelism )); then
        if ! wait "${prewarm_pids[0]}"; then
          prewarm_failures=1
        fi
        prewarm_pids=("${prewarm_pids[@]:1}")
      fi
    done
    for prewarm_pid in "${prewarm_pids[@]}"; do
      if ! wait "${prewarm_pid}"; then
        prewarm_failures=1
      fi
    done
    if [[ "${prewarm_failures}" != "0" ]]; then
      dump_prewarm_diagnostics
      exit 1
    fi

    echo "Installing steady-state ${prewarm_name} DaemonSet"
    "${kubectl_cmd[@]}" apply -n "${prewarm_namespace}" -f - <<EOF
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: ${prewarm_name}
  labels:
    app: ${prewarm_name}
spec:
  selector:
    matchLabels:
      app: ${prewarm_name}
  updateStrategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1
  template:
    metadata:
      labels:
        app: ${prewarm_name}
      annotations:
        art.openpipe/prewarm-token: "${timestamp}-${art_short_sha}"
    spec:
      nodeSelector:
        ${prewarm_node_selector_key}: ${prewarm_node_selector_value}
      imagePullSecrets:
        - name: ${prewarm_image_pull_secret}
      tolerations:
        - operator: Exists
      initContainers:
$(render_prewarm_init_containers | sed 's/^/    /')
      containers:
        - name: pause
          image: registry.k8s.io/pause:3.10
          resources:
            requests:
              cpu: 10m
              memory: 16Mi
$(render_prewarm_host_volumes | sed 's/^/      /')
EOF
    if ! verify_steady_prewarm_daemonset; then
      dump_prewarm_diagnostics
      exit 1
    fi
  fi
  done
else
  echo "Skipping GPU node prewarm"
fi
