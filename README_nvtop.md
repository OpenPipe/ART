# nvtop Installation Script

This script provides an easy way to install nvtop (NVIDIA GPU monitoring tool) on systems where:
- `snap` is not available
- nvtop is not in the default package repositories
- FUSE is not available for AppImage execution

## Quick Installation

```bash
# Make the script executable (if not already)
chmod +x install_nvtop.sh

# Run the installation
./install_nvtop.sh
```

## What the script does

1. **Downloads** the latest nvtop AppImage from GitHub releases
2. **Extracts** the AppImage contents (bypassing FUSE requirements)
3. **Installs** the nvtop binary to `/usr/local/bin/`
4. **Copies** required libraries to `/usr/local/lib/`
5. **Cleans up** temporary files
6. **Tests** the installation

## Requirements

- `wget` (for downloading)
- `sudo` privileges (for system installation)
- x86_64 architecture

## Usage

After installation, simply run:
```bash
nvtop
```

**Controls:**
- `q` - Quit nvtop
- `h` - Show help
- Arrow keys - Navigate

## Troubleshooting

If the installation fails:
1. Check internet connectivity
2. Ensure you have sudo privileges
3. Verify the GitHub release URL is accessible

## Alternative Installation Methods

If this script doesn't work, you can try:
- Building from source (requires cmake, git, build tools)
- Using distro-specific repositories
- Using flatpak (if available)

## Version

Current script installs nvtop v3.1.0 