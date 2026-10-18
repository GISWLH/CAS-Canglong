"""Upload the Space app plus a built asset folder to Hugging Face.

    HF_TOKEN=... python hf_space/scripts/deploy_space.py --build hf_space/build [--space LonghaoWang/CAS-Canglong]

`--build` is the --out folder of build_assets.py (data/ and assets/). The token is only read from the environment.
"""
import argparse
import os
import shutil
import tempfile

from huggingface_hub import HfApi

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = ["app.py", "requirements.txt", "README.md"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", default="LonghaoWang/CAS-Canglong")
    ap.add_argument("--build", required=True)
    ap.add_argument("--message", default="Update CAS-Canglong forecast")
    a = ap.parse_args()
    api = HfApi()
    api.create_repo(a.space, repo_type="space", space_sdk="gradio", exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for f in APP:
            shutil.copy(os.path.join(HERE, f), tmp)
        shutil.copytree(os.path.join(HERE, "scripts"), os.path.join(tmp, "scripts"), ignore=shutil.ignore_patterns("__pycache__"))
        for d in ("data", "assets"):
            shutil.copytree(os.path.join(a.build, d), os.path.join(tmp, d))
        api.upload_folder(folder_path=tmp, repo_id=a.space, repo_type="space", commit_message=a.message,
                          delete_patterns=["data/*", "assets/maps/*"])
    print(f"https://huggingface.co/spaces/{a.space}")


if __name__ == "__main__":
    main()
