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

# Configure git user name and email
git config --global user.name "${GIT_USER_NAME}"
git config --global user.email "${GIT_USER_EMAIL}"
git config --global --add safe.directory /root/sky_workdir

# Reset any uncommitted changes to the last commit
git reset --hard HEAD

# Remove all untracked files and directories
git clean -fd

# Install astral-uv
sudo snap install --classic astral-uv

# Install tmux
apt install tmux -y

# Sync the dependencies
if [ "${INSTALL_EXTRAS:-false}" = "true" ]; then
    uv sync --all-extras
else
    uv sync
fi