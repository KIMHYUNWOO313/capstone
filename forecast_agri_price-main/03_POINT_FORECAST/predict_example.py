"""
TimesFM 2.5 — 농산물 가격 3일 점예측 사용 예제 (Zero-Shot)

배경:
- 4-way 비교 (TimesFM ZS/LoRA + IBM TTM ZS/FT) 결과 ★ TimesFM 2.5 Zero-Shot이 최고
  MASE 0.806 (LoRA 0.811, TTM FT 0.931, TTM ZS 2.184)
- 학습 비용 0, adapter 관리 불필요, 신규 품목 즉시 대응

출력:
- point_forecast: shape (N_items, 3) — 각 품목별 3일 mean 예측값
- quantile_forecast: shape (N_items, 3, 10) — 9 분위 + mean (운영에선 mean 만 사용)

설치 (백엔드 PC):
    pip install timesfm pandas pyarrow numpy
    # GPU 권장 (CUDA 12.x), CPU 도 가능 (배치 추론 5-10배 느림)
"""

import json
import numpy as np
import pandas as pd
import timesfm
from timesfm.configs import ForecastConfig

DATA_PATH    = "../01_DATA/data/full_baseline_extended_20260516.parquet"
META_PATH    = "../01_DATA/meta_baseline.json"
MODEL_ID     = "google/timesfm-2.5-200m-pytorch"   # HuggingFace 에서 자동 다운로드 (≈200MB)
CONTEXT_LEN  = 384                                 # 32 배수, ≈1년치
HORIZON_LEN  = 32                                  # 32 배수, 평가시 앞 3 step만 사용
PRED_LENGTH  = 3                                   # 3일 점예측

# ── 1) 모델 로드 + 컴파일 ──────────────────────────────────────────────────────
print(f"[1/4] loading {MODEL_ID} ...")
model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(MODEL_ID)
model.compile(ForecastConfig(
    max_context=CONTEXT_LEN,
    max_horizon=HORIZON_LEN,
    normalize_inputs=True,      # 모델 내부 RevIN 정규화 사용
    return_backcast=False,      # covariates 미사용 모드
))

# ── 2) 데이터 로드 ─────────────────────────────────────────────────────────────
print("[2/4] loading data ...")
df = pd.read_parquet(DATA_PATH)
if isinstance(df.index, pd.MultiIndex):
    df = df.reset_index()
df = df.sort_values(["item_id", "timestamp"]).reset_index(drop=True)

with open(META_PATH, encoding="utf-8") as f:
    meta = json.load(f)
print(f"  rows: {len(df):,}  items: {df['item_id'].nunique()}  "
      f"period: {df['timestamp'].min().date()} ~ {df['timestamp'].max().date()}")

# ── 3) 추론 (univariate, target 시리즈만 사용) ─────────────────────────────────
# 운영 시: 각 품목별 최근 CONTEXT_LEN 일치 target 시리즈를 numpy 배열로 전달
print("[3/4] running forecast ...")
items, inputs = [], []
for iid, sub in df.groupby("item_id"):
    sub = sub.sort_values("timestamp")
    arr = sub["target"].to_numpy(dtype=np.float32)
    if len(arr) < 64:
        continue
    arr = arr[-CONTEXT_LEN:]    # 최근 CONTEXT_LEN 일치만
    items.append(iid)
    inputs.append(arr)

point_fc, quantile_fc = model.forecast(horizon=PRED_LENGTH, inputs=inputs)
# point_fc: (N, 3)         — 평균 (mean) 점예측값
# quantile_fc: (N, 3, 10)  — [mean, 0.1, 0.2, ..., 0.9] 채널 (운영에선 사용 안 함)

# ── 4) 결과를 DataFrame 으로 정리 ─────────────────────────────────────────────
print("[4/4] formatting output ...")
out_rows = []
for iid, preds in zip(items, point_fc):
    last_ts = df[df["item_id"] == iid]["timestamp"].max()
    for k in range(PRED_LENGTH):
        out_rows.append({
            "item_id": iid,
            "timestamp": last_ts + pd.Timedelta(days=k + 1),
            "y_pred": float(preds[k]),
        })
forecast_df = pd.DataFrame(out_rows)
print(f"\nTotal {len(forecast_df)} predictions (= {len(items)} items x {PRED_LENGTH} days)")
print(forecast_df.head(12).to_string(index=False))

# 특정 품목 예시
item = "apple_fuji_box10kg_high"
sub = forecast_df[forecast_df["item_id"] == item]
print(f"\n{item} - 3-day point forecast (mean):")
print(sub[["timestamp", "y_pred"]].to_string(index=False))
