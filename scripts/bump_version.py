#!/usr/bin/env python3
"""
Version bumping script for ART releases.

Usage:
    python scripts/bump_version.py patch  # 0.3.13 -> 0.3.14
    python scripts/bump_version.py minor  # 0.3.13 -> 0.4.0
    python scripts/bump_version.py major  # 0.3.13 -> 1.0.0
"""

from pathlib import Path
import re
import subprocess
import sys


def get_current_version():
    """Extract current version from pyproject.toml."""
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    content = pyproject_path.read_text()
    match = re.search(r'version = "(\d+\.\d+\.\d+)"', content)
    if not match:
        raise ValueError("Could not find version in pyproject.toml")
    return match.group(1)


def bump_version(current_version, bump_type):
    """Bump version based on type (major, minor, patch)."""
    major, minor, patch = map(int, current_version.split("."))

    if bump_type == "major":
        return f"{major + 1}.0.0"
    elif bump_type == "minor":
        return f"{major}.{minor + 1}.0"
    elif bump_type == "patch":
        return f"{major}.{minor}.{patch + 1}"
    else:
        raise ValueError(f"Invalid bump type: {bump_type}")


def update_version(new_version):
    """Update version in pyproject.toml."""
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    content = pyproject_path.read_text()

    # Update version
    new_content = re.sub(
        r'version = "\d+\.\d+\.\d+"', f'version = "{new_version}"', content
    )

    pyproject_path.write_text(new_content)

    # run uv sync
    subprocess.run(["uv", "sync"])


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ["major", "minor", "patch"]:
        print(__doc__)
        sys.exit(1)

    bump_type = sys.argv[1]

    try:
        current = get_current_version()
        new = bump_version(current, bump_type)

        print(f"Bumping version from {current} to {new}")
        update_version(new)
        print("✓ Updated pyproject.toml")
        print("\nNext steps:")
        print(
            f"1. Commit the change: git add pyproject.toml uv.lock && git commit -m 'Bump version to {new}'"
        )
        print(f"2. Create and push tag: git tag v{new} && git push origin v{new}")
        print(
            "3. The GitHub Action will automatically create a release and publish to PyPI"
        )

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
