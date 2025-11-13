"""
Compatibility shim so `uvicorn main:app` works when run from the BACKEND folder.

The actual FastAPI application lives in app/main.py, but some tooling and
developer workflows expect a top-level `main.py`.  Importing and re-exporting
the FastAPI instance here avoids confusing "Could not import module 'main'"
errors when starting the server.
"""

from app.main import app  # noqa: F401

__all__ = ["app"]
