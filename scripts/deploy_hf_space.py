"""Create/update the Hugging Face Space for this app (called by .github/workflows/deploy-hf.yml).

Environment: HF_TOKEN (required), SPACE_NAME, SMTP_USER / SMTP_PASSWORD (optional).
"""

import os
import shutil
import tempfile

from huggingface_hub import HfApi

FRONT_MATTER = """---
title: DR Explainable AI System
emoji: 👁️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Explainable AI screening for diabetic retinopathy
---

"""

IGNORE = [".git/*", ".github/*", "tests/*", "docs/*", "data/*", "outputs/*", "app/uploads/*",
          "**/__pycache__/*", "*.pyc", ".pytest_cache/*"]


def main():
    token = os.environ["HF_TOKEN"]
    api = HfApi(token=token)
    user = api.whoami()["name"]
    repo_id = f"{user}/{os.environ.get('SPACE_NAME') or 'dr-explainable-ai-system'}"

    api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True)

    # Spaces read their settings from the README front matter
    with tempfile.TemporaryDirectory() as tmp:
        stage = os.path.join(tmp, "space")
        shutil.copytree(".", stage, ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache"))
        with open(os.path.join(stage, "README.md"), encoding="utf-8") as f:
            readme = f.read()
        with open(os.path.join(stage, "README.md"), "w", encoding="utf-8") as f:
            f.write(FRONT_MATTER + readme)

        api.upload_folder(folder_path=stage, repo_id=repo_id, repo_type="space",
                          ignore_patterns=IGNORE,
                          commit_message=f"Deploy {os.environ.get('GITHUB_SHA', 'local')[:7]}")

    for name in ("SMTP_USER", "SMTP_PASSWORD"):
        value = os.environ.get(name)
        if value:
            api.add_space_secret(repo_id, name, value)

    print(f"Deployed: https://huggingface.co/spaces/{repo_id}")
    print(f"App URL:  https://{repo_id.replace('/', '-').replace('_', '-').lower()}.hf.space")


if __name__ == "__main__":
    main()
