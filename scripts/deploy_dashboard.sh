#!/usr/bin/env bash
# scripts/deploy_dashboard.sh
# ----------------------------
# Deploy the Meridian Financial Streamlit dashboard to HuggingFace Spaces.
#
# ACTIVE DEPLOYMENT TARGET: HuggingFace Spaces
# FUTURE/PLANNED TARGET:    AWS ECS / AppRunner (see docs/deployment.md)
#
# Usage:
#   HF_TOKEN=hf_xxx HF_DASH_SPACE=your-username/meridian-dashboard bash scripts/deploy_dashboard.sh
#
# Environment variables:
#   HF_TOKEN        — HuggingFace write token (required)
#   HF_DASH_SPACE   — HF Space repo ID, e.g. bottyash/meridian-dashboard
#   API_BASE_URL    — FastAPI backend URL to bake into the Space
#   COMMIT_MSG      — Optional custom commit message

set -euo pipefail

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
HF_TOKEN="${HF_TOKEN:-}"
HF_DASH_SPACE="${HF_DASH_SPACE:-bottyash/meridian-dashboard}"
API_BASE_URL="${API_BASE_URL:-https://bottyash-meridian-api.hf.space}"
COMMIT_MSG="${COMMIT_MSG:-Deploy dashboard from local}"
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
# Deploy dashboard to HF Spaces
# ---------------------------------------------------------------------------
echo ""
echo "Deploying dashboard to HuggingFace Spaces: $HF_DASH_SPACE"
echo "API_BASE_URL = $API_BASE_URL"

python3 - << PYEOF
import os
from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
space_id = os.environ["HF_DASH_SPACE"]
commit_msg = os.environ.get("COMMIT_MSG", "Deploy dashboard")

# Upload entire dashboard/ folder
api.upload_folder(
    folder_path="dashboard",
    repo_id=space_id,
    repo_type="space",
    ignore_patterns=["*.pyc", "__pycache__", ".pytest_cache"],
    commit_message=commit_msg,
)

# Update the Space secret API_BASE_URL if provided
api_url = os.environ.get("API_BASE_URL", "")
if api_url:
    try:
        api.add_space_secret(repo_id=space_id, key="API_BASE_URL", value=api_url)
        print(f"Space secret API_BASE_URL set to: {api_url}")
    except Exception as e:
        print(f"Could not set API_BASE_URL secret: {e}")

print(f"Dashboard deployed: https://huggingface.co/spaces/{space_id}")
PYEOF

echo ""
echo "Done. Dashboard deployed to: https://huggingface.co/spaces/$HF_DASH_SPACE"
echo "API backend URL configured: $API_BASE_URL"
