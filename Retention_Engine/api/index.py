import sys
from pathlib import Path

# Make the app importable from the Vercel serverless function
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.main import app  # noqa: F401 — Vercel picks up `app`
