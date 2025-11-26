"""
Rate Limiting Middleware
Protects API endpoints from abuse and DDoS attacks
"""

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from typing import Callable

# Initialize rate limiter
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["1000/hour"],  # Default: 1000 requests per hour per IP
    storage_uri="memory://"  # Use in-memory storage (can switch to Redis later)
)

def setup_rate_limiting(app):
    """Setup rate limiting for FastAPI app"""
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    return limiter

# Rate limit decorator for easy use
def rate_limit(limit: str):
    """
    Rate limit decorator
    
    Examples:
        @rate_limit("10/minute")  # 10 requests per minute
        @rate_limit("100/hour")   # 100 requests per hour
        @rate_limit("1000/day")   # 1000 requests per day
    """
    return limiter.limit(limit)

