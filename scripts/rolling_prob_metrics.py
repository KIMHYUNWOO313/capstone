from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import pathlib

import numpy as np
import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor


BASE_Q = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], dtype=float)
ALL_Q = np.array([i / 100 for i in range(1, 100)], dtype=float)
ALL_Q_LABELS = [f"q{int(q * 100):02d}" for q in ALL_Q]


@dataclass
class Paths:
    data_parquet: Path
    meta_json: Path
    model_dir: Path
    output_dir: Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rolling-window WQL/PICP evaluator")
    p.add_argument(
        "--data-parquet",
        default="final_handoff/final_handoff/PROBABILISTIC_FORECAST/data/full_baseline.parquet",
    )
    p.add_argument(
        "--meta-json",
        default="final_handoff/final_handoff/PROBABILISTIC_FORECAST/data/meta_baseline.json",
    )
    p.add_argument(
        "--model-dir",
        default="final_handoff/final_handoff/PROBABILISTIC_FORECAST/model",
    )
    p.add_argument("--model-name", default="Chronos2LoRA_baseline")
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--step-days", type=int, default=15)
    p.add_argument("--output-dir", default="rolling_eval_output")
    return p.parse_args()


def _expand_99_quantiles(base_pred: np.ndarray) -> np.ndarray:
    n = base_pred.shape[0]
    out = np.zeros((n, len(ALL_Q)), dtype=float)
    for i in range(n):
        row = base_pred[i]
        interp = np.interp(ALL_Q, BASE_Q, row)
        left_slope = (row[1] - row[0]) / (BASE_Q[1] - BASE_Q[0])
        right_slope = (row[-1] - row[-2]) / (BASE_Q[-1] - BASE_Q[-2])
        left_mask = ALL_Q < BASE_Q[0]
        right_mask = ALL_Q > BASE_Q[-1]
        interp[left_mask] = row[0] + left_slope * (ALL_Q[left_mask] - BASE_Q[0])
        interp[right_mask] = row[-1] + right_slope * (ALL_Q[right_mask] - BASE_Q[-1])
        out[i] = interp
    return out


def _quantile_losses(y: np.ndarray, q_pred: np.ndarray) -> dict[str, float]:
    denom = float(np.abs(y).sum())
    if denom == 0:
        denom = 1e-9
    out: dict[str, float] = {}
    for idx, q in enumerate(ALL_Q):
        err = y - q_pred[:, idx]
        pinball = np.maximum(q * err, (q - 1.0) * err)
        out[f"WQL@q{int(q*100):02d}"] = float(pinball.sum() / denom)
    out["WQL_overall_q01_q99"] = float(
        np.mean([out[f"WQL@q{int(q*100):02d}"] for q in ALL_Q])
    )
    return out


def _picp_metrics(y: np.ndarray, q_pred: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    for alpha in [50, 60, 70, 80, 90]:
        lower_q = (1 - alpha / 100) / 2
        upper_q = 1 - lower_q
        lower_idx = int(round(lower_q * 100)) - 1
        upper_idx = int(round(upper_q * 100)) - 1
        lower = q_pred[:, lower_idx]
        upper = q_pred[:, upper_idx]
        inside = (y >= lower) & (y <= upper)
        out[f"PICP@{alpha}"] = float(np.mean(inside))
        out[f"MPIW@{alpha}"] = float(np.mean(upper - lower))
    return out


def _build_known_covariates(
    predictor: TimeSeriesPredictor,
    full_reset: pd.DataFrame,
    known_covs: list[str],
    context: TimeSeriesDataFrame,
) -> TimeSeriesDataFrame:
    future_df = predictor.make_future_data_frame(context).reset_index()
    future_df = future_df.merge(
        full_reset[["item_id", "timestamp", *known_covs]],
        on=["item_id", "timestamp"],
        how="left",
    )
    context_reset = context.reset_index()
    last_known = (
        context_reset.groupby("item_id")[known_covs]
        .last()
        .reset_index()
        .rename(columns={col: f"{col}__last" for col in known_covs})
    )
    future_df = future_df.merge(last_known, on="item_id", how="left")
    for col in known_covs:
        future_df[col] = future_df[col].fillna(future_df[f"{col}__last"]).fillna(0)
        future_df = future_df.drop(columns=[f"{col}__last"])
    return TimeSeriesDataFrame.from_data_frame(
        future_df,
        id_column="item_id",
        timestamp_column="timestamp",
    )


def evaluate(paths: Paths, model_name: str, horizon: int, step_days: int) -> None:
    pathlib.WindowsPath = pathlib.PosixPath  # type: ignore[misc, assignment]
    predictor = TimeSeriesPredictor.load(paths.model_dir)
    with paths.meta_json.open(encoding="utf-8") as f:
        meta = json.load(f)
    known_covs = list(meta["known_covariates"])
    train_end = pd.Timestamp(meta["train_end"])

    full_ts = TimeSeriesDataFrame(pd.read_parquet(paths.data_parquet))
    full_reset = full_ts.reset_index().sort_values(["item_id", "timestamp"]).reset_index(drop=True)
    full_actual = full_reset[["item_id", "timestamp", "target"]].rename(columns={"target": "y_true"})
    test_end = pd.Timestamp(full_reset["timestamp"].max())

    start_origin = train_end
    last_origin = test_end - pd.Timedelta(days=horizon)
    origins = list(pd.date_range(start=start_origin, end=last_origin, freq=f"{step_days}D"))
    print(f"rolling_windows={len(origins)} start={origins[0].date()} end={origins[-1].date()}")

    window_rows = []
    pred_rows = []

    for origin in origins:
        context_df = full_reset[full_reset["timestamp"] <= origin]
        context = TimeSeriesDataFrame.from_data_frame(
            context_df,
            id_column="item_id",
            timestamp_column="timestamp",
        )
        known_fc = _build_known_covariates(predictor, full_reset, known_covs, context)
        forecast = pd.DataFrame(
            predictor.predict(context, known_covariates=known_fc, model=model_name).reset_index()
        )
        merged = pd.DataFrame(
            forecast.merge(full_actual, on=["item_id", "timestamp"], how="left")
        ).dropna(subset=["y_true"])
        if merged.empty:
            continue

        base_pred = merged[[f"{q:.1f}" for q in BASE_Q]].to_numpy(dtype=float)
        q99_pred = _expand_99_quantiles(base_pred)
        y = merged["y_true"].to_numpy(dtype=float)

        wql = _quantile_losses(y, q99_pred)
        picp = _picp_metrics(y, q99_pred)
        row = {"origin_date": origin.date().isoformat(), "n_obs": int(len(y)), **wql, **picp}
        window_rows.append(row)

        qdf = pd.DataFrame(q99_pred, columns=ALL_Q_LABELS)
        one_pred = pd.concat(
            [
                merged[["item_id", "timestamp", "y_true"]].reset_index(drop=True),
                qdf.reset_index(drop=True),
            ],
            axis=1,
        )
        one_pred.insert(0, "origin_date", origin.date().isoformat())
        pred_rows.append(one_pred)

    if not window_rows:
        raise RuntimeError("No rolling-window results produced.")

    paths.output_dir.mkdir(parents=True, exist_ok=True)
    per_window = pd.DataFrame(window_rows).sort_values("origin_date")
    all_preds = pd.concat(pred_rows, ignore_index=True)

    agg = {}
    metric_cols = [c for c in per_window.columns if c not in {"origin_date", "n_obs"}]
    for c in metric_cols:
        agg[c] = float(per_window[c].mean())
    summary = pd.DataFrame([agg])

    per_window_path = paths.output_dir / "rolling_metrics_per_window.csv"
    summary_path = paths.output_dir / "rolling_metrics_summary.csv"
    preds_path = paths.output_dir / "rolling_predictions_q01_q99.csv"
    per_window.to_csv(per_window_path, index=False)
    summary.to_csv(summary_path, index=False)
    all_preds.to_csv(preds_path, index=False)

    print(f"saved={per_window_path}")
    print(f"saved={summary_path}")
    print(f"saved={preds_path}")
    show_cols = ["WQL@q50", "WQL@q80", "WQL_overall_q01_q99", "PICP@50", "PICP@80", "PICP@90"]
    print("summary:")
    print(summary[show_cols].to_string(index=False))


def main():
    args = parse_args()
    paths = Paths(
        data_parquet=Path(args.data_parquet),
        meta_json=Path(args.meta_json),
        model_dir=Path(args.model_dir),
        output_dir=Path(args.output_dir),
    )
    evaluate(paths, model_name=args.model_name, horizon=args.horizon, step_days=args.step_days)


if __name__ == "__main__":
    main()
