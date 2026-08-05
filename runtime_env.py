import os
import sys


def ensure_runtime_paths():
    candidates = []

    project_root = os.path.dirname(os.path.abspath(__file__))
    venv_site_packages = os.path.join(project_root, ".venv", "lib", "python3.14", "site-packages")
    if os.path.isdir(venv_site_packages):
        candidates.append(venv_site_packages)

    for path in candidates:
        if path and os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)


ensure_runtime_paths()
