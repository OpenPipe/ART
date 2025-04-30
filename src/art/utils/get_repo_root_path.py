import os


def get_repo_root_path() -> str:
    try:
        # search through parent directories until we find a .git directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        while not os.path.exists(os.path.join(current_dir, ".git")):
            if current_dir == "/":
                raise Exception("Could not find .git directory")
            current_dir = os.path.dirname(current_dir)
        return current_dir
    except Exception:
        return "."
