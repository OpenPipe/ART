#!/bin/bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
cd "${repo_root}"

# Load environment variables from .env file if it exists
if [ -f .env ]; then
    # Read .env file line by line, ignoring comments and empty lines
    while IFS= read -r line || [ -n "$line" ]; do
        # Skip comments and empty lines
        [[ $line =~ ^#.*$ ]] && continue
        [[ -z $line ]] && continue

        key="${line%%=*}"
        current_value="${!key-}"
        if [ -z "${!key+x}" ] ||
            [ -z "${current_value}" ] ||
            { [ "${key}" = "GIT_USER_NAME" ] && [ "${current_value}" = "Your Name" ]; } ||
            { [ "${key}" = "GIT_USER_EMAIL" ] && [ "${current_value}" = "your.email@example.com" ]; } ||
            { [ "${key}" = "INSTALL_EXTRAS" ] && [ "${current_value}" = "false" ]; }; then
            export "$line"
        fi
    done < .env
fi

if ! command -v sudo >/dev/null 2>&1; then
    if [ "$(id -u)" -ne 0 ]; then
        echo "setup requires root or passwordless sudo" >&2
        exit 1
    fi
    sudo_path=/usr/local/bin/sudo
    cat <<'EOF' > "$sudo_path"
#!/bin/sh
if [ "${1:-}" = "-n" ]; then
    shift
fi
exec "$@"
EOF
    chmod +x /usr/local/bin/sudo
fi

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/opt/conda/bin:$PATH"
need_pkgs=()
command -v git >/dev/null 2>&1 || need_pkgs+=("git")
command -v curl >/dev/null 2>&1 || need_pkgs+=("curl")
command -v tmux >/dev/null 2>&1 || need_pkgs+=("tmux")

install_multinode=${INSTALL_MULTINODE:-false}
if [ "$install_multinode" != "true" ] && [ "$install_multinode" != "false" ]; then
    echo "INSTALL_MULTINODE must be true or false" >&2
    exit 1
fi
if [ "${#need_pkgs[@]}" -gt 0 ]; then
    if [ "$(id -u)" -eq 0 ]; then
        apt-get update
        apt-get install -y "${need_pkgs[@]}"
    elif sudo -n true >/dev/null 2>&1; then
        sudo -n apt-get update
        sudo -n apt-get install -y "${need_pkgs[@]}"
    else
        echo "setup requires passwordless sudo to install: ${need_pkgs[*]}" >&2
        exit 1
    fi
fi

# Configure git user name and email
if [ -n "${GIT_USER_NAME:-}" ]; then
    git config --global user.name "${GIT_USER_NAME}"
fi
if [ -n "${GIT_USER_EMAIL:-}" ]; then
    git config --global user.email "${GIT_USER_EMAIL}"
fi
git config --global --add safe.directory "$(pwd)"

if [ "${GIT_RESET_CLEAN:-false}" = "true" ]; then
    # Reset any uncommitted changes to the last commit
    git reset --hard HEAD

    # Remove all untracked files and directories
    git clean -fd
else
    echo "Skipping git reset/clean (GIT_RESET_CLEAN is not true). Preserving synced working tree."
fi

readonly uv_version=0.11.7
if ! uv --version 2>/dev/null | grep -q "^uv ${uv_version} "; then
    curl -LsSf "https://astral.sh/uv/${uv_version}/install.sh" | sh
fi
if ! uv --version; then
    echo "Failed to install uv." >&2
    exit 1
fi

backend_extra=backend
if [ -f /usr/local/cuda/version.json ] &&
    grep -Eq '"version"[[:space:]]*:[[:space:]]*"13\.' /usr/local/cuda/version.json; then
    backend_extra=backend-cu130
fi

if [ "$install_multinode" = "true" ]; then
    if [ "${INSTALL_EXTRAS:-false}" = "true" ]; then
        echo "INSTALL_EXTRAS is incompatible with the Megatron environment" >&2
        exit 1
    fi
    /bin/bash "${repo_root}/src/art/megatron/setup.sh"
else
    sync_extras=(--extra "$backend_extra")
    if [ "${INSTALL_EXTRAS:-false}" = "true" ]; then
        sync_extras+=(--extra tinker --extra langgraph --extra plotting)
    fi
    uv sync "${sync_extras[@]}" --frozen
fi
