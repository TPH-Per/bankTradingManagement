# This file is created to support the uvicorn main:app command
# It imports the app from app.py to maintain compatibility

# Import app from app.py in the same directory
# We need to handle both cases: running from BACKEND/app and from BACKEND
import sys
import os
from pathlib import Path

# Get the directory containing this file
current_dir = Path(__file__).parent.resolve()

# Add parent directory to path to allow importing app as a package
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Import app from the app package
from app.app import app
from fastapi.responses import Response, RedirectResponse

# Add a simple favicon route to prevent 404 errors
@app.get("/favicon.ico")
async def favicon():
    # Return an empty response with 204 status (No Content)
    return Response(status_code=204)

# Add a root route with service info (no redirect)
@app.get("/")
async def root():
    return {
        "service": "bankTrading ML Service",
        "version": "2.3",
        "status": "online",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "transactions": "/rt/transactions",
            "predictions": {
                "all": "/ml/predict/all",
                "cash_in": "/ml/predict/cash-in",
                "cash_out": "/ml/predict/cash-out"
            },
            "model_status": "/ml/status"
        }
    }

# This allows running: uvicorn main:app --host 0.0.0.0 --port 8000 --reload