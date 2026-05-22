#!/usr/bin/env bash
# scripts/deploy_backend.sh
# --------------------------
# Deploy the Meridian Financial FastAPI backend to HuggingFace Spaces.
#
# ACTIVE DEPLOYMENT TARGET: HuggingFace Spaces
# FUTURE/PLANNED TARGET:    AWS EC2 (see docs/deployment.md#future-aws-architecture)
#
# Usage:
#   HF_TOKEN=hf_xxx HF_API_SPACE=your-username/meridian-api bash scripts/deploy_backend.sh
#
# Environment variables:
#   HF_TOKEN        — HuggingFace write token (required)
#   HF_API_SPACE    — HF Space repo ID, e.g. bottyash/meridian-api
#   COMMIT_MSG      — Optional custom commit message

set -euo pipefail

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
HF_TOKEN="${HF_TOKEN:-}"
HF_API_SPACE="${HF_API_SPACE:-bottyash/meridian-api}"
COMMIT_MSG="${COMMIT_MSG:-Deploy backend from local}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
if [[ -z "$HF_TOKEN" ]]; then
  echo "ERROR: HF_TOKEN environment variable is required."
  echo "  export HF_TOKEN=hf_your_token_here"
  exit 1
fi

command -v python3 >/dev/null 2>&1 || { echo "ERROR: python3 not found"; exit 1; }
python3 -c "import huggingface_hub" 2>/dev/null || pip install huggingface_hub -q

# ---------------------------------------------------------------------------
# Run tests before deploy
# ---------------------------------------------------------------------------
echo "Running test suite before deploy..."
cd "$REPO_ROOT"
python3 -m pytest tests/ -q --tb=short
echo "Tests passed."

# ---------------------------------------------------------------------------
# Deploy to HF Spaces
# ---------------------------------------------------------------------------
echo ""
echo "Deploying backend to HuggingFace Spaces: $HF_API_SPACE"

python3 - << PYEOF
import os
from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
space_id = os.environ["HF_API_SPACE"]
commit_msg = os.environ.get("COMMIT_MSG", "Deploy backend")

# Upload src/ and config/
for folder, path_in_repo in [("src", "src"), ("config", "config")]:
    api.upload_folder(
        folder_path=folder,
        repo_id=space_id,
        repo_type="space",
        path_in_repo=path_in_repo,
        ignore_patterns=["*.pyc", "__pycache__"],
        commit_message=commit_msg,
    )

# Upload runtime files
for file, path_in_repo in [
    ("Dockerfile", "Dockerfile"),
    ("requirements.txt", "requirements.txt"),
]:
    api.upload_file(
        path_or_fileobj=file,
        path_in_repo=path_in_repo,
        repo_id=space_id,
        repo_type="space",
        commit_message=commit_msg,
    )

print(f"Backend deployed: https://huggingface.co/spaces/{space_id}")
PYEOF

echo ""
echo "Done. Backend deployed to: https://huggingface.co/spaces/$HF_API_SPACE"
