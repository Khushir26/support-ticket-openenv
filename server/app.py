"""
server/app.py — Support Ticket Agent OpenEnv Server

Required entry point for `openenv validate`.
Rules:
  - Function must be named `main` (not start)
  - Must have if __name__ == '__main__': block
  - pyproject.toml must have: server = "server.app:main"
"""
from __future__ import annotations

import os
import sys

# Add project root to path so server/ can import from root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Re-export the FastAPI app from root main.py
from main import app  # noqa: F401

__all__ = ["app", "main"]


def main() -> None:
    """Entry point called by `uv run server` and [project.scripts]."""
    import uvicorn
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "7860")),
        workers=1,
        reload=False,
    )


if __name__ == "__main__":
    main()
