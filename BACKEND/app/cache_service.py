"""
Redis Caching Service
Provides caching layer for improved performance and reduced database load
"""

import os
import json
import hashlib
from functools import wraps
from typing import Optional, Callable, Any
import logging

logger = logging.getLogger(__name__)

# Try to import Redis
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis not available. Install with: pip install redis")

# Redis client (initialized on first use)
_redis_client: Optional[redis.Redis] = None


def get_redis_client() -> Optional[redis.Redis]:
    """Get or create Redis client"""
    global _redis_client
    
    if not REDIS_AVAILABLE:
        return None
    
    if _redis_client is None:
        try:
            redis_host = os.getenv("REDIS_HOST", "localhost")
            redis_port = int(os.getenv("REDIS_PORT", "6379"))
            redis_db = int(os.getenv("REDIS_DB", "0"))
            
            _redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            # Test connection
            _redis_client.ping()
            logger.info(f"Redis connected: {redis_host}:{redis_port}")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Caching disabled.")
            _redis_client = None
    
    return _redis_client


def cache_result(ttl: int = 300, key_prefix: str = ""):
    """
    Cache decorator for API endpoints
    
    Args:
        ttl: Time to live in seconds
        key_prefix: Prefix for cache keys
    
    Usage:
        @cache_result(ttl=3600, key_prefix="ml")
        async def get_prediction(features):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            client = get_redis_client()
            if not client:
                # Redis not available, execute function directly
                return await func(*args, **kwargs)
            
            # Generate cache key
            cache_key = _generate_cache_key(func.__name__, key_prefix, args, kwargs)
            
            try:
                # Try to get from cache
                cached = client.get(cache_key)
                if cached:
                    logger.debug(f"Cache HIT: {cache_key}")
                    return json.loads(cached)
                
                # Cache miss - execute function
                logger.debug(f"Cache MISS: {cache_key}")
                result = await func(*args, **kwargs)
                
                # Store in cache
                client.setex(cache_key, ttl, json.dumps(result, default=str))
                return result
            except Exception as e:
                logger.warning(f"Cache error for {cache_key}: {e}. Executing function directly.")
                return await func(*args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            client = get_redis_client()
            if not client:
                return func(*args, **kwargs)
            
            cache_key = _generate_cache_key(func.__name__, key_prefix, args, kwargs)
            
            try:
                cached = client.get(cache_key)
                if cached:
                    logger.debug(f"Cache HIT: {cache_key}")
                    return json.loads(cached)
                
                logger.debug(f"Cache MISS: {cache_key}")
                result = func(*args, **kwargs)
                client.setex(cache_key, ttl, json.dumps(result, default=str))
                return result
            except Exception as e:
                logger.warning(f"Cache error for {cache_key}: {e}. Executing function directly.")
                return func(*args, **kwargs)
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


def _generate_cache_key(func_name: str, prefix: str, args: tuple, kwargs: dict) -> str:
    """Generate cache key from function name and arguments"""
    # Create hash of arguments
    key_data = f"{func_name}:{str(args)}:{str(sorted(kwargs.items()))}"
    key_hash = hashlib.md5(key_data.encode()).hexdigest()[:16]
    
    if prefix:
        return f"{prefix}:{func_name}:{key_hash}"
    return f"cache:{func_name}:{key_hash}"


def invalidate_cache(pattern: str):
    """
    Invalidate cache entries matching pattern
    
    Args:
        pattern: Redis key pattern (e.g., "ml:*", "account:*")
    
    Example:
        invalidate_cache("ml:*")  # Invalidate all ML prediction caches
        invalidate_cache("account:123:*")  # Invalidate all caches for account 123
    """
    client = get_redis_client()
    if not client:
        return
    
    try:
        keys = client.keys(pattern)
        if keys:
            client.delete(*keys)
            logger.info(f"Invalidated {len(keys)} cache entries matching {pattern}")
    except Exception as e:
        logger.warning(f"Error invalidating cache: {e}")


def clear_all_cache():
    """Clear all cache entries"""
    client = get_redis_client()
    if not client:
        return
    
    try:
        client.flushdb()
        logger.info("All cache cleared")
    except Exception as e:
        logger.warning(f"Error clearing cache: {e}")

