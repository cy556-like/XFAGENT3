import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path=".",
    repo_id="yinchen16/company-doc-agent",
    repo_type="space",
    token="REDACTED_HF_TOKEN",
    ignore_patterns=["venv", "__pycache__", ".git", "data/chroma_db", "*.pyc", ".env"],
)
