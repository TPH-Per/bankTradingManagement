# Ingestion & API Experiments

## Ingestion (memory mode)
- Requests: 30
- Avg latency: 59.01 ms | p95: 67.12 ms
- Daily CSV rows: 20 | Unique client_tx_id: 20
- Dedup in daily CSV: present

## Prediction API
- Latency (cash_in): 50.61 ms
- Latency (cash_out): 72.77 ms
- Latency (all): 88.99 ms
- Non-negative outputs: True

## Input flexibility & direction inference
- Negative amount infers cash_out: True (Δcash_out=123000)

## Notes
- This test runs with Cassandra disabled (in-memory fallback).
- Deduplication at DB level (client_tx_dedup) is not exercised here; daily CSV-level dedup is validated.
