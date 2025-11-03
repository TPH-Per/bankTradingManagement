CREATE KEYSPACE IF NOT EXISTS pldt_rt
  WITH REPLICATION = {'class':'SimpleStrategy','replication_factor':1};

CREATE KEYSPACE IF NOT EXISTS pldt_audit
  WITH REPLICATION = {'class':'SimpleStrategy','replication_factor':1};

CREATE KEYSPACE IF NOT EXISTS pldt_rt
  WITH REPLICATION = {'class':'NetworkTopologyStrategy','datacenter1':3};

CREATE KEYSPACE IF NOT EXISTS pldt_audit
  WITH REPLICATION = {'class':'NetworkTopologyStrategy','datacenter1':3};

USE pldt_rt;

CREATE TABLE IF NOT EXISTS tx_by_account_day (
  account_id text,
  event_date date,          -- ngày UTC tính từ event_ts
  event_ts   timestamp,     -- thời điểm giao dịch (UTC, naive trong Cassandra)
  tx_id      uuid,          -- id server side (idempotent có thể map từ client_tx_id)
  amount     decimal,
  currency   text,
  merchant   text,
  status     text,          -- PENDING/SETTLED/REVERSED/...
  extra_json text,          -- JSON mở rộng (mô tả, stock code, ...)

  PRIMARY KEY ((account_id, event_date), event_ts, tx_id)
) WITH CLUSTERING ORDER BY (event_ts DESC, tx_id ASC)
  AND compaction = {
    'class': 'TimeWindowCompactionStrategy',
    'compaction_window_unit': 'DAYS',
    'compaction_window_size': '1'
  };

-- SELECT account_id, event_date, event_ts, tx_id, amount, currency, merchant, status, extra_json
-- FROM pldt_rt.tx_by_account_day
-- WHERE account_id=? AND event_date=? LIMIT ?;

CREATE TABLE IF NOT EXISTS client_tx_dedup (
  account_id    text,
  client_tx_id  text,      -- idempotency key do client gửi
  tx_id         uuid,      -- tx_id thực sự (server phát sinh)
  created_at    timestamp,
  PRIMARY KEY ((account_id), client_tx_id)
);
-- Khi ghi: INSERT IF NOT EXISTS; nếu đã tồn tại thì dùng lại tx_id cũ.

CREATE TABLE IF NOT EXISTS kpi_daily (
  event_date date,     -- ngày UTC
  metric     text,     -- tên KPI (total_amount, num_tx, avg_ticket, ...)
  value      double,
  updated_at timestamp,
  PRIMARY KEY ((event_date), metric)
);

-- SELECT metric, value, updated_at FROM pldt_rt.kpi_daily WHERE event_date=?;
--
-- INSERT INTO pldt_rt.kpi_daily (event_date, metric, value, updated_at)
-- VALUES (?, ?, ?, toTimestamp(now()));

CREATE TABLE IF NOT EXISTS kpi_daily_by_account (
  account_id text,
  event_date date,
  metric     text,
  value      double,
  updated_at timestamp,
  PRIMARY KEY ((account_id, event_date), metric)
);

CREATE TABLE IF NOT EXISTS api_calls (
  day        date,        -- ngày UTC từ ts
  ts         timestamp,   -- thời điểm gọi
  endpoint   text,        -- /rt/transactions, /kpi/daily/..., /ml/m5p/...
  method     text,        -- GET/POST/...
  status     int,         -- 200/400/500...
  account_id text,        -- nếu có
  client_ip  text,
  extra_json text,        -- payload rút gọn / lỗi / tham số
  PRIMARY KEY ((day), ts, endpoint)
) WITH CLUSTERING ORDER BY (ts DESC);

-- 1) Thử ghi vào dedup:
INSERT INTO pldt_rt.client_tx_dedup (account_id, client_tx_id, tx_id, created_at)
VALUES (?, ?, ?, toTimestamp(now()))
IF NOT EXISTS;

-- 2) Nếu áp dụng (applied=true) → tx_id này là mới, tiếp tục INSERT vào tx_by_account_day.
--    Nếu không áp dụng → SELECT tx_id từ client_tx_dedup và trả id cũ (không ghi trùng).

-- Chọn keyspace
USE pldt_rt;

-- Ghi 1 giao dịch
INSERT INTO tx_by_account_day (
  account_id, event_date, event_ts, tx_id, amount, currency, merchant, status, extra_json
) VALUES (
  'ACC001', '2025-10-12', toTimestamp(now()), now(), 123.45, 'USD', 'ONLINE', 'SETTLED', '{"desc":"demo"}'
);

-- Đọc theo tài khoản + ngày
SELECT * FROM tx_by_account_day
WHERE account_id='ACC001' AND event_date='2025-10-12' LIMIT 5;

-- KPI ngày
INSERT INTO kpi_daily (event_date, metric, value, updated_at)
VALUES ('2025-10-12', 'total_amount', 12345.67, toTimestamp(now()));

SELECT * FROM kpi_daily WHERE event_date='2025-10-12';

-- Audit
USE pldt_audit;
INSERT INTO api_calls (day, ts, endpoint, method, status, account_id, client_ip, extra_json)
VALUES ('2025-10-12', toTimestamp(now()), '/healthz', 'GET', 200, null, '127.0.0.1', null);

SELECT * FROM api_calls WHERE day='2025-10-12' LIMIT 3;



CREATE KEYSPACE IF NOT EXISTS pldt_rt
  WITH REPLICATION = {'class':'NetworkTopologyStrategy','datacenter1':3};

USE pldt_rt;

CREATE TABLE IF NOT EXISTS tx_by_account_day (
  account_id  text,
  event_date  date,        -- ngày UTC (partition cùng account)
  event_ts    timestamp,   -- thời điểm giao dịch
  tx_id       uuid,
  amount      decimal,
  currency    text,
  merchant    text,
  status      text,
  extra_json  text,
  PRIMARY KEY ((account_id, event_date), event_ts, tx_id)
) WITH CLUSTERING ORDER BY (event_ts DESC, tx_id ASC);

CREATE TABLE IF NOT EXISTS api_calls (
  day        date,        -- ngày UTC từ ts
  ts         timestamp,   -- thời điểm gọi
  endpoint   text,        -- /rt/transactions, /kpi/daily/..., /ml/m5p/...
  method     text,        -- GET/POST/...
  status     int,         -- 200/400/500...
  account_id text,        -- nếu có
  client_ip  text,
  extra_json text,        -- payload rút gọn / lỗi / tham số
  PRIMARY KEY ((day), ts, endpoint)
) WITH CLUSTERING ORDER BY (ts DESC);


USE pldt_audit;
-- nên chỉ rõ keyspace để tránh nhầm, ví dụ pldt_rt
SELECT *
FROM pldt_rt.tx_by_account_day


