"""
Prometheus Monitoring Metrics
Provides metrics for system monitoring and observability
"""

import time
from typing import Optional
from prometheus_client import Counter, Histogram, Gauge, generate_latest, REGISTRY
from fastapi import Request
from fastapi.responses import Response

# Try to import prometheus_client
try:
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

if PROMETHEUS_AVAILABLE:
    # API Metrics
    api_requests_total = Counter(
        'api_requests_total',
        'Total number of API requests',
        ['method', 'endpoint', 'status_code']
    )
    
    api_request_duration = Histogram(
        'api_request_duration_seconds',
        'API request duration in seconds',
        ['endpoint'],
        buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0)
    )
    
    # Business Metrics
    transactions_total = Counter(
        'transactions_total',
        'Total number of transactions',
        ['type', 'status']
    )
    
    ml_predictions_total = Counter(
        'ml_predictions_total',
        'Total number of ML predictions',
        ['model_name', 'type']
    )
    
    # System Metrics
    active_connections = Gauge(
        'active_connections',
        'Number of active connections'
    )
    
    cassandra_health = Gauge(
        'cassandra_health',
        'Cassandra health status (1=healthy, 0=unhealthy)'
    )
    
    hdfs_health = Gauge(
        'hdfs_health',
        'HDFS health status (1=healthy, 0=unhealthy)'
    )
    
    ml_model_accuracy = Gauge(
        'ml_model_accuracy',
        'ML model prediction accuracy',
        ['model_name']
    )
    
    # Cache Metrics
    cache_hits_total = Counter(
        'cache_hits_total',
        'Total cache hits',
        ['cache_key']
    )
    
    cache_misses_total = Counter(
        'cache_misses_total',
        'Total cache misses',
        ['cache_key']
    )
else:
    # Dummy metrics if prometheus_client not available
    api_requests_total = None
    api_request_duration = None
    transactions_total = None
    ml_predictions_total = None
    active_connections = None
    cassandra_health = None
    hdfs_health = None
    ml_model_accuracy = None
    cache_hits_total = None
    cache_misses_total = None


def track_request(request: Request, response, duration: float):
    """Track API request metrics"""
    if not PROMETHEUS_AVAILABLE or not api_requests_total:
        return
    
    try:
        endpoint = request.url.path
        method = request.method
        status_code = response.status_code
        
        # Track request count
        api_requests_total.labels(
            method=method,
            endpoint=endpoint,
            status_code=status_code
        ).inc()
        
        # Track request duration
        if api_request_duration:
            api_request_duration.labels(endpoint=endpoint).observe(duration)
    except Exception:
        pass  # Don't break the request if metrics fail


def track_transaction(tx_type: str, status: str):
    """Track transaction metrics"""
    if transactions_total:
        transactions_total.labels(type=tx_type, status=status).inc()


def track_ml_prediction(model_name: str, prediction_type: str):
    """Track ML prediction metrics"""
    if ml_predictions_total:
        ml_predictions_total.labels(
            model_name=model_name,
            type=prediction_type
        ).inc()


def update_cassandra_health(is_healthy: bool):
    """Update Cassandra health metric"""
    if cassandra_health:
        cassandra_health.set(1 if is_healthy else 0)


def update_hdfs_health(is_healthy: bool):
    """Update HDFS health metric"""
    if hdfs_health:
        hdfs_health.set(1 if is_healthy else 0)


def update_ml_accuracy(model_name: str, accuracy: float):
    """Update ML model accuracy metric"""
    if ml_model_accuracy:
        ml_model_accuracy.labels(model_name=model_name).set(accuracy)


def track_cache_hit(cache_key: str):
    """Track cache hit"""
    if cache_hits_total:
        cache_hits_total.labels(cache_key=cache_key).inc()


def track_cache_miss(cache_key: str):
    """Track cache miss"""
    if cache_misses_total:
        cache_misses_total.labels(cache_key=cache_key).inc()


def get_metrics_response() -> Response:
    """Get Prometheus metrics endpoint response"""
    if not PROMETHEUS_AVAILABLE:
        return Response(
            content="# Prometheus client not installed\n",
            media_type="text/plain"
        )
    
    return Response(
        content=generate_latest(REGISTRY),
        media_type="text/plain"
    )

