# 📊 Cấu trúc dữ liệu Training Dataset

## ✅ Dataset hiện tại đã được cập nhật

### File: `data/cash_daily_train_realistic.csv`

**Số lượng:** 365 dòng (1 năm dữ liệu từ 2023-01-01 đến 2023-12-31)

### Cấu trúc Columns:

```csv
date,cash_in,cash_out,channel,balance,day_of_week,month,quarter
2023-01-01,3704000,2726000,POS,978000.0,6,1,1
2023-01-02,6762000,2563000,ATM,5177000.0,0,1,1
...
```

| Column | Type | Description | Ví dụ |
|--------|------|-------------|-------|
| `date` | string | Ngày (YYYY-MM-DD) | 2023-01-01 |
| `cash_in` | float | Tiền vào trong ngày (VND) | 3704000 |
| `cash_out` | float | Tiền ra trong ngày (VND) | 2726000 |
| `channel` | string | Kênh giao dịch | ATM, ONLINE, POS, TRANSFER |
| `balance` | float | Số dư tích lũy (VND) | 978000.0 |
| `day_of_week` | int | Thứ (0=Thứ 2, 6=Chủ nhật) | 6 |
| `month` | int | Tháng (1-12) | 1 |
| `quarter` | int | Quý (1-4) | 1 |

## 📈 Đặc điểm dữ liệu

### ✅ Điểm mạnh:
1. **365 ngày đầy đủ** - Đủ để train models cho:
   - next_day (1 ngày)
   - h7_sum (7 ngày)
   - next_month_sum (30 ngày)

2. **Xu hướng thực tế:**
   - Cuối tuần thấp hơn (~60-70% ngày thường)
   - Ngày payday (15, cuối tháng) cao hơn (~150%)
   - Tăng dần qua năm (~30% growth)

3. **4 Channels đa dạng:**
   - ATM: 40%
   - ONLINE: 30%
   - POS: 20%
   - TRANSFER: 10%

4. **Không có số âm** - Tất cả cash_in và cash_out đều dương

### 📊 Thống kê:
```
Cash In:  Min=2,582,000  Max=16,886,000  Avg=7,074,633 VND
Cash Out: Min=1,778,000  Max=6,351,000   Avg=3,423,595 VND
```

## 🤔 Câu hỏi: Có nên chia theo tháng không?

### ❌ **KHÔNG nên chia theo tháng vì:**

1. **Models cần dữ liệu liên tục:**
   - h7_sum cần 7 ngày liên tiếp
   - next_month_sum cần 30 ngày liên tiếp
   - Chia theo tháng sẽ gây gaps

2. **Rolling window bị vỡ:**
   - `lag7_in` cần data 7 ngày trước
   - Nếu chia file, dữ liệu tháng 2 không có lag từ tháng 1

3. **Training kém hiệu quả:**
   - Models học patterns qua thời gian
   - Cần thấy xu hướng dài hạn

### ✅ **NÊN giữ 1 file duy nhất với:**

```
data/
└── cash_daily_train_realistic.csv  (365+ rows)
```

## 🔄 Cập nhật dữ liệu thế nào?

### Phương án 1: Append vào file hiện tại (Khuyên dùng)
```python
# Khi có giao dịch mới, service tự động append
# File: app.py → _update_training_dataset_from_transaction()

# Dữ liệu sẽ được append theo ngày:
# 2023-12-31  → existing
# 2024-01-01  → appended
# 2024-01-02  → appended
```

**Ưu điểm:**
- ✅ Liên tục, không gaps
- ✅ Rolling features hoạt động tốt
- ✅ Tự động retrain sau mỗi transaction

### Phương án 2: Định kỳ generate lại (Cho dev/test)
```bash
# Chạy script để generate dataset mới
cd ml_service
python generate_realistic_data.py
```

## 📝 Thêm dữ liệu thực tế

### Nếu bạn có dữ liệu từ database:

```python
import pandas as pd

# Query từ database
df = pd.read_sql("""
    SELECT
        DATE(event_ts) as date,
        SUM(CASE WHEN transaction_type = 'cash_in' THEN amount ELSE 0 END) as cash_in,
        SUM(CASE WHEN transaction_type = 'cash_out' THEN amount ELSE 0 END) as cash_out,
        channel
    FROM transactions
    WHERE event_ts >= '2023-01-01'
    GROUP BY DATE(event_ts), channel
    ORDER BY date
""", conn)

# Save
df.to_csv('data/cash_daily_train_realistic.csv', index=False)
```

### Merge với dữ liệu hiện có:

```python
# Load existing
existing = pd.read_csv('data/cash_daily_train_realistic.csv')

# Load new data
new_data = pd.read_csv('new_transactions.csv')

# Merge by date and aggregate
combined = pd.concat([existing, new_data])
combined = combined.groupby('date').agg({
    'cash_in': 'sum',
    'cash_out': 'sum',
    'channel': 'first'  # or 'most_common'
}).reset_index()

# Sort and save
combined = combined.sort_values('date')
combined.to_csv('data/cash_daily_train_realistic.csv', index=False)
```

## 🚀 Train Models với dataset mới

### Via GUI:
1. Mở http://localhost:8000 hoặc gui.html
2. Tab "🤖 Mô hình ML"
3. File name: `cash_daily_train_realistic.csv`
4. Click "Huấn luyện mô hình"

### Via API:
```bash
curl -X POST "http://localhost:8000/ml/m5p/train" \
  -H "Content-Type: application/json" \
  -d '{
    "data_file_path": "./data/cash_daily_train_realistic.csv"
  }'
```

## ⚠️ Lưu ý quan trọng

1. **Minimum rows cần thiết:**
   - next_day models: 30+ rows
   - h7_sum models: 40+ rows (30 + 7 + buffer)
   - next_month_sum models: 90+ rows (30 + 30 + buffer)

2. **File phải có columns:**
   - `date` (required)
   - `cash_in` (required)
   - `cash_out` (required)
   - `channel` (optional, default "DEFAULT")

3. **Data quality:**
   - Không có gaps trong dates
   - Không có NULL values
   - Không có số âm
   - Format date: YYYY-MM-DD

## 📦 Backup và Version Control

### Backup định kỳ:
```bash
# Backup mỗi tháng
cp data/cash_daily_train_realistic.csv \
   data/backups/cash_daily_$(date +%Y%m).csv
```

### Git (nếu dùng):
```bash
# Add to .gitignore nếu data nhạy cảm
echo "data/*.csv" >> .gitignore

# Hoặc commit nếu OK
git add data/cash_daily_train_realistic.csv
git commit -m "Update training dataset"
```

---

**Tóm lại:**
- ✅ **1 file duy nhất** với 365+ dòng
- ✅ **Tự động append** khi có giao dịch mới
- ❌ **KHÔNG chia** theo tháng
- ✅ Dataset hiện tại **ĐÃ SẴN SÀNG** để train!
