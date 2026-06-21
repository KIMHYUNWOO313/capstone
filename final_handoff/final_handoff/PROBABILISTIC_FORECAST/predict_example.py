"""
Chronos2 LoRA — 농산물 가격 10일 확률범위 예측 사용 예제

출력 컬럼: [item_id, timestamp, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, mean]
  0.1 / 0.9 → 80% 예측 구간 하한 / 상한
  0.2 / 0.8 → 60% 예측 구간 하한 / 상한
  0.5       → 중앙값 (점예측 대용)
"""

import json
import pandas as pd
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame

MODEL_DIR   = "model/"
DATA_PATH   = "data/full_baseline.parquet"
META_PATH   = "data/meta_baseline.json"
LORA_MODEL  = "Chronos2LoRA_baseline"

# ── 모델 & 메타 로드 ──────────────────────────────────────────────────────────
predictor = TimeSeriesPredictor.load(MODEL_DIR)

with open(META_PATH, encoding="utf-8") as f:
    meta = json.load(f)
KNOWN_COVS = meta["known_covariates"]

# ── 데이터 로드 ───────────────────────────────────────────────────────────────
full       = TimeSeriesDataFrame(pd.read_parquet(DATA_PATH))
full_reset = full.reset_index()

# ── known covariate 프레임 생성 ───────────────────────────────────────────────
def make_known(context: TimeSeriesDataFrame) -> TimeSeriesDataFrame:
    """
    예측 구간(10일) known covariate 프레임 생성.
    실제 서비스에서는 기상청 예보·공휴일 캘린더·한국은행 API 등으로 값을 채워야 함.
    여기서는 full 데이터의 lookup 테이블로 대체 (평가용).
    결측 시 마지막 관측값으로 forward-fill.
    """
    fd = predictor.make_future_data_frame(context).reset_index()
    fd = fd.merge(full_reset[["item_id", "timestamp"] + KNOWN_COVS],
                  on=["item_id", "timestamp"], how="left")
    for c in KNOWN_COVS:
        if fd[c].isna().any():
            fd[c] = fd.groupby("item_id")[c].transform(
                lambda s: s.ffill().bfill()).fillna(0)
    return TimeSeriesDataFrame.from_data_frame(
        fd, id_column="item_id", timestamp_column="timestamp")

# ── 예측 실행 ────────────────────────────────────────────────────────────────
context  = full
known_fc = make_known(context)

forecast = predictor.predict(context, known_covariates=known_fc, model=LORA_MODEL)

# ── 결과 출력 ────────────────────────────────────────────────────────────────
fc = forecast.reset_index()
print(fc.head(20).to_string(index=False))

# 특정 품목 예시
item = "apple_fuji_box10kg_high"
print(f"\n{item} - 10-day forecast (80% interval):")
print(forecast.loc[item][["0.1", "0.5", "0.9"]].to_string())
