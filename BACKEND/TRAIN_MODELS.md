# Hướng dẫn Train Multi-Target Cash Flow Models

## 📋 Tổng quan

Hệ thống sử dụng **6 mô hình M5P độc lập** để dự đoán cash flow:

### Models:
1. **cash_in_next_day** - Dự đoán tiền vào ngày tiếp theo
2. **cash_out_next_day** - Dự đoán tiền ra ngày tiếp theo
3. **cash_in_h7_sum** - Tổng tiền vào 7 ngày tiếp theo
4. **cash_out_h7_sum** - Tổng tiền ra 7 ngày tiếp theo
5. **cash_in_next_month_sum** - Tổng tiền vào tháng tiếp theo (30 ngày)
6. **cash_out_next_month_sum** - Tổng tiền ra tháng tiếp theo (30 ngày)

## 📊 Chuẩn bị Dataset

### Format CSV cần có:
```csv
date,cash_in,cash_out,channel
2024-01-01,5000000,3000000,ATM
2024-01-02,5200000,3100000,ONLINE
2024-01-03,4800000,2900000,ATM
...
```

### Columns tối thiểu:
- `date` - Ngày (YYYY-MM-DD)
- `cash_in` - Tiền vào trong ngày (VND)
- `cash_out` - Tiền ra trong ngày (VND)
- `channel` (optional) - Kênh giao dịch

### Đặt file vào:
```
ml_service/app/data/cash_daily_train_realistic.csv
```

## 🚀 Cách 1: Train qua GUI

### Bước 1: Khởi động service
```bash
cd ml_service/app
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Bước 2: Mở GUI
Mở trình duyệt: `http://localhost:8000` hoặc mở file `gui.html`

### Bước 3: Train models
1. Chuyển sang tab **"🤖 Mô hình ML"**
2. Scroll xuống section **"Huấn luyện mô hình ML"**
3. Nhập tên file: `cash_daily_train_realistic.csv`
4. Click **"Huấn luyện mô hình"**
5. Đợi ~1-2 phút cho đến khi hoàn thành

### Kết quả:
- 6 models sẽ được train và save vào thư mục `models/`
- Hiển thị metrics: RMSE, R², MAE
- Models tự động load khi restart service

## 🔧 Cách 2: Train qua API

### Train tất cả 6 models:
```bash
curl -X POST "http://localhost:8000/ml/m5p/train" \
  -H "Content-Type: application/json" \
  -d '{
    "data_file_path": "./data/cash_daily_train_realistic.csv"
  }'
```

### Response:
```json
{
  "rmse": 125000.45,
  "r2": 0.925,
  "mae": 98000.32,
  "model_path": "models/cash_in_next_day/m5p_cash_in_next_day.pkl",
  "target_column": "cash_in_next_day"
}
```

## 🔍 Kiểm tra trạng thái models

### Qua API:
```bash
curl http://localhost:8000/ml/status
```

### Response:
```json
{
  "ready": true,
  "models": {
    "cash_in_next_day": true,
    "cash_out_next_day": true,
    "cash_in_h7": true,
    "cash_out_h7": true,
    "cash_in_next_month": true,
    "cash_out_next_month": true
  },
  "version": "2.3"
}
```

## 💡 Tips

### Auto-bootstrap khi khởi động:
Models tự động load từ disk khi service khởi động nếu đã có sẵn.

### Force retrain:
Đặt biến môi trường để force train lại:
```bash
export M5P_FORCE_RETRAIN=1
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Thay đổi dataset path:
```bash
export M5P_DATASET=/path/to/your/data.csv
uvicorn main:app --host 0.0.0.0 --port 8000
```

## 📈 Test Predictions

### Test Cash In predictions:
```bash
curl -X POST "http://localhost:8000/ml/predict/cash-in" \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "cash_in_d0": 5000000,
      "cash_out_d0": 3000000,
      "cash_net_d0": 2000000,
      "lag1_in": 4800000,
      "lag7_in": 5200000,
      "roll_mean_7_in": 5100000,
      "dow": 1,
      "is_weekend": 0,
      "is_month_end": 0,
      "is_payday": 0,
      "channel": "ATM"
    }
  }'
```

### Response:
```json
{
  "next_day": 5100000.0,
  "h7_sum": 35700000.0,
  "next_month_sum": 153000000.0
}
```

### Test Cash Out predictions:
```bash
curl -X POST "http://localhost:8000/ml/predict/cash-out" \
  -H "Content-Type: application/json" \
  -d '{
    "features": { ... same as above ... }
  }'
```

### Response:
```json
{
  "next_day": 3200000.0,
  "h7_sum": 22400000.0,
  "next_month_sum": 96000000.0
}
```

## ⚠️ Troubleshooting

### Lỗi: "Dataset not found"
- Kiểm tra file tồn tại tại `ml_service/app/data/cash_daily_train_realistic.csv`
- Hoặc dùng đường dẫn tuyệt đối

### Lỗi: "Missing required columns"
- Dataset phải có: `date`, `cash_in`, `cash_out`
- Format date phải là YYYY-MM-DD

### Models không train
- Kiểm tra logs: Xem terminal console
- Đảm bảo có đủ dữ liệu (tối thiểu 30 rows)

### Performance kém
- Cần ít nhất 90+ ngày dữ liệu để train tốt
- Dữ liệu phải liên tục, không gaps lớn

## 📁 Cấu trúc Models sau khi train

```
models/
├── cash_in_next_day/
│   ├── m5p_cash_in_next_day.pkl
│   └── schema_cash_in_next_day.json
├── cash_out_next_day/
│   ├── m5p_cash_out_next_day.pkl
│   └── schema_cash_out_next_day.json
├── cash_in_h7/
│   ├── m5p_cash_in_h7.pkl
│   └── schema_cash_in_h7.json
├── cash_out_h7/
│   ├── m5p_cash_out_h7.pkl
│   └── schema_cash_out_h7.json
├── cash_in_next_month/
│   ├── m5p_cash_in_next_month.pkl
│   └── schema_cash_in_next_month.json
└── cash_out_next_month/
    ├── m5p_cash_out_next_month.pkl
    └── schema_cash_out_next_month.json
```

Mỗi model có 2 files:
- `.pkl` - Model đã train (pickle)
- `.json` - Schema với feature names và metadata

---

**Thành công!** 🎉 Bây giờ bạn có thể dự đoán cash flow cho ngày/tuần/tháng tiếp theo!
