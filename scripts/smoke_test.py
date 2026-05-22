"""
scripts/smoke_test.py
---------------------
Lightweight smoke test for the Meridian Financial serving API.

Tests
-----
1. GET /health        — status == "ok"
2. GET /metrics       — required keys present
3. GET /openapi.json  — all 6 endpoints registered

Can be run:
  * During Docker build verification
  * As a CI post-deploy check
  * Manually against any running instance

Usage
-----
  python scripts/smoke_test.py [--base-url http://localhost:8000]

Exit codes
----------
  0  all tests passed
  1  one or more tests failed
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request


# ---------------------------------------------------------------------------
# Smoke test cases
# ---------------------------------------------------------------------------

def _get(url: str, timeout: int = 10) -> dict:
    """HTTP GET with a basic error wrapper."""
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def test_health(base: str) -> tuple[bool, str]:
    try:
        data = _get(f"{base}/health")
        assert data.get("status") == "ok", f"Expected status=ok, got {data}"
        return True, "status == ok"
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def test_metrics(base: str) -> tuple[bool, str]:
    try:
        data = _get(f"{base}/metrics")
        required = {"total_requests", "uptime_seconds", "prediction_distribution", "rag_retrieval_stats"}
        missing = required - set(data.keys())
        assert not missing, f"Missing keys: {missing}"
        return True, f"keys present: {sorted(required)}"
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def test_openapi(base: str) -> tuple[bool, str]:
    try:
        data = _get(f"{base}/openapi.json")
        paths = set(data.get("paths", {}).keys())
        required_paths = {
            "/health", "/predict", "/ask-complaints",
            "/batch-score", "/customer-intel", "/metrics",
        }
        missing = required_paths - paths
        assert not missing, f"Missing endpoints: {missing}"
        return True, f"all {len(required_paths)} endpoints registered"
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

TESTS = [
    ("GET /health", test_health),
    ("GET /metrics", test_metrics),
    ("GET /openapi.json", test_openapi),
]


def run_smoke_tests(base_url: str) -> int:
    """Run all smoke tests and return exit code (0=pass, 1=fail)."""
    base = base_url.rstrip("/")
    print(f"\nSmoke tests against {base}")
    print("=" * 60)

    passed = 0
    failed = 0

    for name, test_fn in TESTS:
        ok, msg = test_fn(base)
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}  —  {msg}")
        if ok:
            passed += 1
        else:
            failed += 1

    print("=" * 60)
    print(f"Result: {passed} passed, {failed} failed\n")
    return 0 if failed == 0 else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Meridian Financial API smoke tests")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Base URL of the running API (default: http://localhost:8000)",
    )
    args = parser.parse_args()
    sys.exit(run_smoke_tests(args.base_url))


if __name__ == "__main__":
    main()
