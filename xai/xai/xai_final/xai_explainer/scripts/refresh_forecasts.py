"""두 모델을 실행해 xai_explainer/inputs/forecast_*.parquet 갱신.

이 스크립트는 final_handoff/PROBABILISTIC_FORECAST/ 와 POINT_FORECAST/ 가
바로 위에 있어야 동작하며, autogluon.timeseries / timesfm 환경이 필요합니다.
팀원에게 보낼 때는 미리 한 번 실행해 inputs/에 결과를 동봉하세요.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PKG_DIR = Path(__file__).resolve().parent.parent
HANDOFF_DIR = PKG_DIR.parent
PROB_DIR = HANDOFF_DIR / "PROBABILISTIC_FORECAST"
POINT_DIR = HANDOFF_DIR / "POINT_FORECAST"
INPUT_DIR = PKG_DIR / "inputs"


def refresh_chronos2() -> None:
    sys.path.insert(0, str(PROB_DIR))
    from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

    predictor = TimeSeriesPredictor.load(str(PROB_DIR / "model"))
    with open(PROB_DIR / "data" / "meta_baseline.json", encoding="utf-8") as f:
        meta = json.load(f)
    known_covs = meta["known_covariates"]

    full = TimeSeriesDataFrame(pd.read_parquet(PROB_DIR / "data" / "full_baseline.parquet"))
    full_reset = full.reset_index()

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

    forecast = predictor.predict(full, known_covariates=known_fc, model="Chronos2LoRA_baseline")
    out = forecast.reset_index()
    out_path = INPUT_DIR / "forecast_10day.parquet"
    out.to_parquet(out_path, index=False)
    print(f"saved: {out_path}  (rows={len(out)})")


def refresh_timesfm() -> None:
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

    df = pd.read_parquet(POINT_DIR / "data" / "full_baseline.parquet")
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
    print(f"saved: {out_path}  (rows={len(out)})")


def main():
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[1/2] Chronos2 LoRA 10일 확률예측...")
    refresh_chronos2()
    print("[2/2] TimesFM 2.5 3일 점예측...")
    refresh_timesfm()


if __name__ == "__main__":
    main()
