# Experimental Results
Dataset: `/mnt/c/temp/aa/ml_service/data/cash_daily_train_realistic.csv`
Rows: 733 | Date range: 2023-01-01 → 2025-11-04

## Training
Total time: 5.36s

- cash_in_next_day: rmse=1831420.77 r2=0.7719 mae=1287140.83 time=1.67s
- cash_out_next_day: rmse=629260.27 r2=0.7697 mae=478121.10 time=1.03s
- cash_in_h7: rmse=11868161.37 r2=0.6352 mae=8760016.77 time=0.70s
- cash_out_h7: rmse=2754503.43 r2=0.8576 mae=2059562.04 time=0.46s
- cash_in_next_month: rmse=19000977.36 r2=0.8850 mae=14638475.20 time=0.81s
- cash_out_next_month: rmse=7822489.47 r2=0.9108 mae=5683936.03 time=0.69s

## Inference latency (CPU)
- predict_all: 57.728 ms/call
- predict_cash_in: 41.871 ms/call
- predict_cash_out: 25.856 ms/call
