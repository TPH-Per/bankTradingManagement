# This file is created to support the uvicorn main:app command
# It imports the app from app.py to maintain compatibility

from app import app
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