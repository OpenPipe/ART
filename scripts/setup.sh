#!/bin/bash

# Load environment variables from .env file if it exists
if [ -f .env ]; then
    # Read .env file line by line, ignoring comments and empty lines
    while IFS= read -r line || [ -n "$line" ]; do
        # Skip comments and empty lines
        [[ $line =~ ^#.*$ ]] && continue
        [[ -z $line ]] && continue
        
        # Export the variable
        export "$line"
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

need_pkgs=()
command -v git >/dev/null 2>&1 || need_pkgs+=("git")
command -v curl >/dev/null 2>&1 || need_pkgs+=("curl")
command -v tmux >/dev/null 2>&1 || need_pkgs+=("tmux")

if [ "${#need_pkgs[@]}" -gt 0 ]; then
    apt-get update
    apt-get install -y "${need_pkgs[@]}"
fi

# Configure git user name and email
git config --global user.name "${GIT_USER_NAME}"
git config --global user.email "${GIT_USER_EMAIL}"
git config --global --add safe.directory /root/sky_workdir

if [ "${GIT_RESET_CLEAN:-true}" = "true" ]; then
    # Reset any uncommitted changes to the last commit
    git reset --hard HEAD

    # Remove all untracked files and directories
    git clean -fd
else
    echo "Skipping git reset/clean (GIT_RESET_CLEAN is not true). Preserving synced working tree."
fi

# Install astral-uv
if ! command -v uv >/dev/null 2>&1; then
    if ! curl -LsSf https://astral.sh/uv/install.sh | sh; then
        echo "Failed to install uv." >&2
        exit 1
    fi
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

# Update uv
uv self update

# Sync the dependencies
if [ "${INSTALL_EXTRAS:-false}" = "true" ]; then
    uv sync --all-extras
else
    uv sync --extra backend
fi