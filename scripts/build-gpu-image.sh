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
  --no-prewarm-nodes     Skip pre-pulling the pushed image on GPU nodes
  --pull-image-repo REPO Image repository for cluster pulls/prewarm
  --prewarm-timeout DUR  Timeout for the prewarm DaemonSet rollout (default: 30m)
  --tag TAG              Image tag to publish
  --help                 Show this help
EOF
}

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

cluster_name=""
infra="${SKY_INFRA:-k8s/cks-wb3}"
image_repo="${ART_IMAGE_REPO:-}"
pull_image_repo="${ART_PULL_IMAGE_REPO:-}"
image_tag=""
docker_config_path="${DOCKER_CONFIG_PATH:-${HOME}/.docker/config.json}"
buildkit_image="${BUILDKIT_IMAGE:-moby/buildkit:v0.29.0-rootless}"
buildkit_namespace="${KUBECTL_NAMESPACE:-default}"
buildkit_wait_timeout="${BUILDKIT_WAIT_TIMEOUT:-300s}"
no_cache="${NO_CACHE:-false}"
prewarm_nodes="${PREWARM_NODES:-true}"
prewarm_namespace="${PREWARM_NAMESPACE:-default}"
prewarm_name="${PREWARM_NAME:-art-gpu-image-prewarm}"
prewarm_image_pull_secret="${PREWARM_IMAGE_PULL_SECRET:-art-gpu-registry-auth}"
prewarm_node_selector="${PREWARM_NODE_SELECTOR:-node.coreweave.cloud/class=gpu}"
prewarm_timeout="${PREWARM_TIMEOUT:-30m}"

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
    --no-prewarm-nodes)
      prewarm_nodes=false
      shift
      ;;
    --pull-image-repo)
      pull_image_repo="$2"
      shift 2
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

case "${infra}" in
  k8s/*)
    kube_context="${infra#k8s/}"
    ;;
  kubernetes/*)
    kube_context="${infra#kubernetes/}"
    ;;
  *)
    echo "Unsupported --infra '${infra}'. Use k8s/<kubectl-context>." >&2
    exit 1
    ;;
esac

kubectl_cmd=(kubectl --context "${kube_context}")
if [[ "${prewarm_node_selector}" != *=* ]]; then
  echo "PREWARM_NODE_SELECTOR must be a key=value selector, got: ${prewarm_node_selector}" >&2
  exit 1
fi
prewarm_node_selector_key="${prewarm_node_selector%%=*}"
prewarm_node_selector_value="${prewarm_node_selector#*=}"

art_sha="$(git -C "${repo_root}" rev-parse HEAD)"
art_short_sha="$(git -C "${repo_root}" rev-parse --short=12 HEAD)"
timestamp="$(date +%m%d-%H%M%S)"

if [[ -z "${image_tag}" ]]; then
  image_tag="skypilot-${art_short_sha}"
fi

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
  "${kubectl_cmd[@]}" delete pod -n "${buildkit_namespace}" "${cluster_name}" \
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
prewarm_image="${pull_image_repo}:${image_tag}"
if [[ -n "${image_digest}" ]]; then
  prewarm_image="${pull_image_repo}@${image_digest}"
fi

if [[ "${prewarm_nodes}" == "true" ]]; then
  gpu_node_count="$("${kubectl_cmd[@]}" get nodes -l "${prewarm_node_selector}" --no-headers 2>/dev/null | wc -l | tr -d ' ')"
  if [[ "${gpu_node_count}" == "0" ]]; then
    echo "Skipping GPU node prewarm: no nodes match ${prewarm_node_selector}"
  else
    echo "Prewarming ${prewarm_image} on ${gpu_node_count} GPU node(s)"
    "${kubectl_cmd[@]}" create secret generic "${prewarm_image_pull_secret}" \
      -n "${prewarm_namespace}" \
      --from-file=.dockerconfigjson="${registry_auth_json_path}" \
      --type=kubernetes.io/dockerconfigjson \
      --dry-run=client -o yaml \
      | "${kubectl_cmd[@]}" apply -n "${prewarm_namespace}" -f -
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
      maxUnavailable: 100%
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
        - name: prepull
          image: ${prewarm_image}
          imagePullPolicy: Always
          command: ["bash", "-lc", "true"]
          resources:
            requests:
              cpu: 10m
              memory: 16Mi
      containers:
        - name: pause
          image: registry.k8s.io/pause:3.10
          resources:
            requests:
              cpu: 10m
              memory: 16Mi
EOF
    "${kubectl_cmd[@]}" rollout status -n "${prewarm_namespace}" "daemonset/${prewarm_name}" --timeout="${prewarm_timeout}"
  fi
else
  echo "Skipping GPU node prewarm"
fi
