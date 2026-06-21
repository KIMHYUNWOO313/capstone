"""두 모델을 실행해 xai_explainer/inputs/forecast_*.parquet 갱신.

이 스크립트는 final_handoff 구조 (02_PROBABILISTIC_FORECAST, 03_POINT_FORECAST,
01_DATA, 05_WEATHER_FORECAST) 가 형제 폴더로 있어야 동작하며, autogluon.timeseries
+ timesfm 환경이 모두 필요합니다. 팀원에게 보낼 때는 미리 한 번 실행해 inputs/에
결과를 동봉하세요.

기본 모드:
  - Chronos2 LoRA: 기상청 일기예보 기반 known_covariates (forecast 모드)
  - TimesFM 2.5: univariate zero-shot (변경 없음)

CLI:
  python refresh_forecasts.py              # 기본 (forecast 모드)
  python refresh_forecasts.py --lookup     # Chronos2 known 을 lookup 으로 (오프라인 평가용)
  python refresh_forecasts.py --only chronos2|timesfm   # 단일 모델만 갱신
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PKG_DIR = Path(__file__).resolve().parent.parent          # 04_XAI_EXPLAINER/
HANDOFF_DIR = PKG_DIR.parent                              # final_handoff/
PROB_DIR = HANDOFF_DIR / "02_PROBABILISTIC_FORECAST"
POINT_DIR = HANDOFF_DIR / "03_POINT_FORECAST"
DATA_DIR = HANDOFF_DIR / "01_DATA" / "data"
META_PATH = HANDOFF_DIR / "01_DATA" / "meta_baseline.json"
WF_DIR = HANDOFF_DIR / "05_WEATHER_FORECAST"
INPUT_DIR = PKG_DIR / "inputs"

DATA_PATH = DATA_DIR / "full_baseline_extended_20260516.parquet"


def refresh_chronos2(mode: str = "forecast") -> None:
    """Chronos2 LoRA 10일 확률예측 갱신.

    mode:
      - "forecast" (기본): 기상청 단기/중기 일기예보 기반 known_covariates
      - "lookup" : full_baseline 의 실제값 lookup (테스트셋 평가용)
    """
    sys.path.insert(0, str(PROB_DIR))
    sys.path.insert(0, str(WF_DIR))
    from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

    predictor = TimeSeriesPredictor.load(str(PROB_DIR / "model"))
    with open(META_PATH, encoding="utf-8") as f:
        meta = json.load(f)
    known_covs = meta["known_covariates"]
    item_station_map = meta["item_station_map"]

    full = TimeSeriesDataFrame(pd.read_parquet(DATA_PATH))
    full_reset = full.reset_index()

    if mode == "forecast":
        from covariate_builder import build_known_covariates_frame
        known_df = build_known_covariates_frame(full_reset, item_station_map, horizon=10)
        known_df = known_df[["item_id", "timestamp"] + known_covs]
        known_fc = TimeSeriesDataFrame.from_data_frame(
            known_df, id_column="item_id", timestamp_column="timestamp"
        )
        print(f"  [chronos2] mode=forecast known_rows={len(known_df)}")
    else:
        fd = predictor.make_future_data_frame(full).reset_index()
        fd = fd.merge(
            full_reset[["item_id", "timestamp"] + known_covs],
            on=["item_id", "timestamp"], how="left",
        )
        for c in known_covs:
            if fd[c].isna().any():
                fd[c] = fd.groupby("item_id")[c].transform(
                    lambda s: s.ffill().bfill()
                ).fillna(0)
        known_fc = TimeSeriesDataFrame.from_data_frame(
            fd, id_column="item_id", timestamp_column="timestamp"
        )
        print(f"  [chronos2] mode=lookup known_rows={len(fd)}")

    forecast = predictor.predict(full, known_covariates=known_fc, model="Chronos2LoRA_baseline")
    out = forecast.reset_index()
    out_path = INPUT_DIR / "forecast_10day.parquet"
    out.to_parquet(out_path, index=False)
    print(f"  saved: {out_path}  (rows={len(out)})")


def refresh_timesfm() -> None:
    """TimesFM 2.5 ZS univariate 3일 점예측 갱신."""
    sys.path.insert(0, str(POINT_DIR))
    import timesfm
    from timesfm.configs import ForecastConfig

    CONTEXT_LEN = 384
    HORIZON_LEN = 32
    PRED_LENGTH = 3
    MODEL_ID = "google/timesfm-2.5-200m-pytorch"

    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(MODEL_ID)
    model.compile(ForecastConfig(
        max_context=CONTEXT_LEN,
        max_horizon=HORIZON_LEN,
        normalize_inputs=True,
        return_backcast=False,
    ))

    df = pd.read_parquet(DATA_PATH)
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    df = df.sort_values(["item_id", "timestamp"]).reset_index(drop=True)

    items, inputs = [], []
    for iid, sub in df.groupby("item_id"):
        sub = sub.sort_values("timestamp")
        arr = sub["target"].to_numpy(dtype=np.float32)
        if len(arr) < 64:
            continue
        items.append(iid)
        inputs.append(arr[-CONTEXT_LEN:])

    point_fc, _ = model.forecast(horizon=PRED_LENGTH, inputs=inputs)

    rows = []
    for iid, preds in zip(items, point_fc):
        last_ts = df[df["item_id"] == iid]["timestamp"].max()
        for k in range(PRED_LENGTH):
            rows.append({
                "item_id": iid,
                "timestamp": last_ts + pd.Timedelta(days=k + 1),
                "y_pred": float(preds[k]),
            })
    out = pd.DataFrame(rows)
    out_path = INPUT_DIR / "forecast_3day.parquet"
    out.to_parquet(out_path, index=False)
    print(f"  saved: {out_path}  (rows={len(out)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookup", action="store_true",
                    help="Chronos2 known 을 lookup 방식으로 (오프라인 평가용)")
    ap.add_argument("--only", choices=["chronos2", "timesfm"], default=None,
                    help="단일 모델만 갱신")
    args = ap.parse_args()

    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    mode = "lookup" if args.lookup else "forecast"

    if args.only in (None, "chronos2"):
        print(f"[1/2] Chronos2 LoRA 10일 확률예측  (mode={mode})...")
        refresh_chronos2(mode=mode)
    if args.only in (None, "timesfm"):
        print("[2/2] TimesFM 2.5 ZS 3일 점예측...")
        refresh_timesfm()
    print("[done] xai_explainer/inputs/ 갱신 완료.")


if __name__ == "__main__":
    main()
