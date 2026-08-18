#!/bin/bash
set -eo pipefail

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
    sudo_path="/usr/local/bin/sudo"
    if [ ! -w /usr/local/bin ]; then
        sudo_path="$HOME/.local/bin/sudo"
        mkdir -p "$HOME/.local/bin"
        export PATH="$HOME/.local/bin:$PATH"
    fi

    cat <<'EOF' > "$sudo_path"
#!/bin/sh
exec "$@"
EOF
    chmod +x "$sudo_path"
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
    apt-get update
    apt-get install -y "${need_pkgs[@]}"
fi

# Configure git user name and email
git config --global user.name "${GIT_USER_NAME}"
git config --global user.email "${GIT_USER_EMAIL}"
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
    scripts/setup_multinode.sh
    export HYBRID_EP_MULTINODE=1
    export USE_NIXL=1
    export NIXL_HOME=/usr/local/art-multinode/nixl
    export UCX_HOME=/usr/local/art-multinode/ucx
    export LD_LIBRARY_PATH="$NIXL_HOME/lib/x86_64-linux-gnu:$UCX_HOME/lib:${LD_LIBRARY_PATH:-}"
    /bin/bash src/art/megatron/setup.sh
else
    sync_extras=(--extra "$backend_extra")
    if [ "${INSTALL_EXTRAS:-false}" = "true" ]; then
        sync_extras+=(--extra tinker --extra langgraph --extra plotting)
    fi
    uv sync "${sync_extras[@]}" --frozen
fi
