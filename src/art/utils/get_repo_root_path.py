import os


def get_repo_root_path() -> str:
    # search through parent directories until we find a .git directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    while not os.path.exists(os.path.join(current_dir, ".git")):
        current_dir = os.path.dirname(current_dir)
    return current_dir
