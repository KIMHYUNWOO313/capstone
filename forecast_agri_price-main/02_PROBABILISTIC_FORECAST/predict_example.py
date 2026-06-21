"""
Chronos2 LoRA — 농산물 가격 10일 확률범위 예측 사용 예제 (운영 + 평가 듀얼 모드)

출력 컬럼: [item_id, timestamp, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, mean]
  0.1 / 0.9 → 80% 예측 구간 하한 / 상한
  0.2 / 0.8 → 60% 예측 구간 하한 / 상한
  0.5       → 중앙값 (점예측 대용)

모드:
  - "forecast" (기본, 운영용)
        기상청 단기예보(D+1~D+3) + 중기예보(D+4~D+10) 로 weather_temp_range 채움.
        나머지 4종 known은 마지막 관측치 ffill.
        실패 시 자동으로 "lookup" 으로 폴백.
  - "lookup"  (평가/오프라인)
        full_baseline 의 실제값을 lookup 해 채움 (테스트셋 평가용).

CLI:
  python predict_example.py [forecast|lookup]
"""

import json
import sys
from pathlib import Path

import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

# ── 경로 ──────────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "model"
DATA_PATH = HERE.parent / "01_DATA/data/full_baseline_extended_20260516.parquet"
META_PATH = HERE.parent / "01_DATA/meta_baseline.json"
WF_DIR = HERE.parent / "05_WEATHER_FORECAST"

LORA_MODEL = "Chronos2LoRA_baseline"

# 05_WEATHER_FORECAST 모듈 import (기상청 API 기반 known_covariates 빌더)
sys.path.insert(0, str(WF_DIR))

# ── 모델 & 메타 로드 ──────────────────────────────────────────────────────────
predictor = TimeSeriesPredictor.load(str(MODEL_DIR))

with open(META_PATH, encoding="utf-8") as f:
    meta = json.load(f)
KNOWN_COVS = meta["known_covariates"]
ITEM_STATION_MAP = meta["item_station_map"]

# ── 데이터 로드 ───────────────────────────────────────────────────────────────
full = TimeSeriesDataFrame(pd.read_parquet(DATA_PATH))
full_reset = full.reset_index()


# ── known covariate 프레임 생성 (운영용: 기상청 일기예보 기반) ────────────────
def make_known_from_forecast(context: TimeSeriesDataFrame) -> TimeSeriesDataFrame:
    """기상청 단기/중기 일기예보 → weather_temp_range D+1~D+10, 그 외 4종 ffill.

    호출: 05_WEATHER_FORECAST.covariate_builder.build_known_covariates_frame
    """
    from covariate_builder import KNOWN_COVS as WF_KNOWN_COVS
    from covariate_builder import build_known_covariates_frame

    full_df = context.reset_index() if isinstance(context.index, pd.MultiIndex) else context.copy()
    known_df = build_known_covariates_frame(full_df, ITEM_STATION_MAP, horizon=10)
    # AutoGluon 이 기대하는 컬럼 순서/세트로 정렬
    known_df = known_df[["item_id", "timestamp"] + WF_KNOWN_COVS]
    return TimeSeriesDataFrame.from_data_frame(
        known_df, id_column="item_id", timestamp_column="timestamp"
    )


# ── known covariate 프레임 생성 (평가용: full_baseline lookup) ────────────────
def make_known_from_lookup(context: TimeSeriesDataFrame) -> TimeSeriesDataFrame:
    """full_baseline 의 실제 미래값을 lookup. 테스트셋 평가나 오프라인 검증용."""
    fd = predictor.make_future_data_frame(context).reset_index()
    fd = fd.merge(
        full_reset[["item_id", "timestamp"] + KNOWN_COVS],
        on=["item_id", "timestamp"], how="left",
    )
    for c in KNOWN_COVS:
        if fd[c].isna().any():
            fd[c] = fd.groupby("item_id")[c].transform(
                lambda s: s.ffill().bfill()).fillna(0)
    return TimeSeriesDataFrame.from_data_frame(
        fd, id_column="item_id", timestamp_column="timestamp"
    )


def make_known(context: TimeSeriesDataFrame, mode: str = "forecast") -> TimeSeriesDataFrame:
    """운영: 기상청 일기예보, 평가: lookup. API 실패 시 lookup 자동 fallback."""
    if mode == "lookup":
        return make_known_from_lookup(context)
    try:
        kf = make_known_from_forecast(context)
        # 길이 검증 (46품목 × 10일 = 460행)
        if len(kf) == 0:
            raise RuntimeError("known frame empty (API 응답 부족 추정)")
        print(f"[make_known] mode=forecast rows={len(kf)}  "
              f"items={kf.reset_index()['item_id'].nunique()}")
        return kf
    except Exception as e:
        print(f"[make_known] forecast 모드 실패: {e!r} → lookup 폴백")
        return make_known_from_lookup(context)


# ── 진입점 ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "forecast"
    assert mode in ("forecast", "lookup"), f"invalid mode: {mode}"

    print(f"[predict_example] mode={mode}  data={DATA_PATH.name}")
    context = full
    known_fc = make_known(context, mode=mode)

    forecast = predictor.predict(context, known_covariates=known_fc, model=LORA_MODEL)

    fc = forecast.reset_index()
    print(f"\n총 {len(fc)} 행 예측  ({fc['item_id'].nunique()} 품목 × {len(fc) // max(fc['item_id'].nunique(), 1)} 일)")
    print(fc.head(12).to_string(index=False))

    item = "apple_fuji_box10kg_high"
    print(f"\n{item} - 10-day forecast (80% interval):")
    print(forecast.loc[item][["0.1", "0.5", "0.9"]].to_string())
