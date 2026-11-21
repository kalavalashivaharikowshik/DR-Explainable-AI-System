"""Make sure the ONNX model is real, not a Git LFS pointer.

Some hosts (and clones made without `git lfs`) only receive the small LFS
pointer file. In that case download the real file from GitHub's LFS media
endpoint. Used by the Dockerfile; safe to run any time.
"""

import os
import sys
import urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPO = os.environ.get("DR_GITHUB_REPO", "kalavalashivaharikowshik/DR-Explainable-AI-System")
BRANCH = os.environ.get("DR_GITHUB_BRANCH", "main")
FILES = ["models/trained/best_model.onnx"]


def is_pointer(path):
    with open(path, "rb") as f:
        return f.read(64).startswith(b"version https://git-lfs")


def main():
    for rel in FILES:
        path = os.path.join(ROOT, rel)
        if os.path.exists(path) and not is_pointer(path):
            print(f"ok: {rel} ({os.path.getsize(path) / 1e6:.1f} MB)")
            continue
        url = f"https://media.githubusercontent.com/media/{REPO}/{BRANCH}/{rel}"
        print(f"downloading {rel} from {url}")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".part"
        urllib.request.urlretrieve(url, tmp)
        if is_pointer(tmp) or os.path.getsize(tmp) < 1_000_000:
            os.remove(tmp)
            sys.exit(f"error: could not download the real {rel}")
        os.replace(tmp, path)
        print(f"ok: {rel} ({os.path.getsize(path) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
