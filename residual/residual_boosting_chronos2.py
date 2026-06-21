# -*- coding: utf-8 -*-
"""
residual_boosting_chronos2.py
==============================
Chronos-2 zero-shot 예측의 잔차(residual)를 LightGBM으로 학습해 보정하는
Residual Boosting 파이프라인.

전체 93개 item에 대해:
  1. train 마지막 14일을 자체 test로 분리
  2. 나머지 train 구간에서 rolling 잔차 수집 → LightGBM 학습
  3. Chronos-2 base 예측 + LightGBM 잔차 보정
  4. 보정 전/후 지표 비교 + TimesFM 스타일 시각화
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm

from chronos.chronos2.pipeline import Chronos2Pipeline as TSFMPipeline

# ==========================================
# CONFIG
# ==========================================
BASE_DIR = Path(__file__).resolve().parent

CONFIG = {
    "model_id":          "amazon/chronos-2",
    "train_path":        str(BASE_DIR / "production_split" / "production_split" / "train_data.csv"),
    "test_path":         str(BASE_DIR / "production_split" / "production_split" / "test_data.csv"),
    "context_length":    365,
    "prediction_length": 14,
    "rolling_stride":    56,
    "past_plot_days":    90,
    "lgb_n_rounds":      500,
    "lgb_early_stop":    50,
    "lgb_lr":            0.05,
}

OUTPUT_DIR = str(BASE_DIR / "residual_boost_output")
PLOT_DIR   = os.path.join(OUTPUT_DIR, "plots")

FUTURE_COV_COLS = [
    "month", "temp_avg", "rain_sum", "humid_avg", "sunshine_sum",
    "temp_diff", "dayofweek", "weekofyear", "month_sin", "month_cos",
    "dow_sin", "dow_cos", "weather_index", "rain_impact",
]
PAST_COV_COLS = [
    "oil_diesel", "cpi_total", "gov_bond_3y", "epu", "m2_sa",
    "price_diff", "price_ma7", "temp_rolling_mean_7",
    "oil_diesel_lag_1", "oil_diesel_lag_3", "temp_avg_lag_1", "rain_sum_lag_1",
]
ALL_FEATURE_COLS = FUTURE_COV_COLS + PAST_COV_COLS


# ==========================================
# 유틸
# ==========================================
def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    mask = actual != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def _chronos_forecast_full(pipeline, context: np.ndarray, pred_len: int):
    """전체 샘플(quantile용) + median 반환."""
    pred_input = {"target": context}
    result = pipeline.predict([pred_input], prediction_length=pred_len)
    samples = list(result)[0].numpy()[0]          # (num_samples, pred_len)
    median  = np.median(samples, axis=0)           # (pred_len,)
    return median, samples


# ==========================================
# 1) train 구간 rolling 잔차 수집
# ==========================================
def collect_train_residuals(
    pipeline,
    train_df: pd.DataFrame,
    item_ids: list[str],
) -> pd.DataFrame:
    ctx      = CONFIG["context_length"]
    pred_len = CONFIG["prediction_length"]
    stride   = CONFIG["rolling_stride"]

    rows: list[dict] = []
    for item_id in item_ids:
        df_item    = train_df[train_df["item_id"] == item_id].sort_values("date").reset_index(drop=True)
        target_arr = df_item["target"].values.astype(float)
        n = len(target_arr)

        origins = list(range(ctx, n - pred_len + 1, stride))
        if not origins:
            continue

        print(f"  [{item_id}] rolling windows: {len(origins)}", flush=True)
        for origin in tqdm(origins, desc=item_id, leave=False):
            context        = target_arr[:origin]
            actual_segment = target_arr[origin : origin + pred_len]
            forecast_arcsinh, _ = _chronos_forecast_full(pipeline, context, pred_len)
            residuals = actual_segment - forecast_arcsinh

            for h in range(pred_len):
                idx = origin + h
                if idx >= n:
                    break
                row = {"item_id": item_id, "horizon": h + 1,
                       "residual": residuals[h], "chronos_pred": forecast_arcsinh[h]}
                for col in ALL_FEATURE_COLS:
                    if col in df_item.columns:
                        row[col] = df_item[col].iat[idx]
                rows.append(row)

    return pd.DataFrame(rows)


# ==========================================
# 2) LightGBM 잔차 모델 학습
# ==========================================
def train_residual_model(residual_df: pd.DataFrame) -> lgb.Booster:
    feat_cols = [c for c in ALL_FEATURE_COLS if c in residual_df.columns]
    feat_cols += ["horizon", "chronos_pred"]

    X = residual_df[feat_cols].copy()
    y = residual_df["residual"].values
    ds = lgb.Dataset(X, label=y)

    params = {
        "objective": "regression", "metric": "mae",
        "learning_rate": CONFIG["lgb_lr"], "num_leaves": 63,
        "min_child_samples": 20, "subsample": 0.8,
        "colsample_bytree": 0.8, "verbosity": -1,
    }
    print(f"\n[LightGBM] Training on {len(X)} rows, {len(feat_cols)} features ...", flush=True)
    booster = lgb.train(
        params, ds, num_boost_round=CONFIG["lgb_n_rounds"],
        valid_sets=[ds],
        callbacks=[lgb.log_evaluation(100), lgb.early_stopping(CONFIG["lgb_early_stop"])],
    )
    return booster


# ==========================================
# 3) 전체 item 평가 + TimesFM 스타일 시각화
# ==========================================
def evaluate_all_items(
    pipeline, booster: lgb.Booster, full_df: pd.DataFrame,
):
    ctx      = CONFIG["context_length"]
    pred_len = CONFIG["prediction_length"]
    past_n   = CONFIG["past_plot_days"]
    os.makedirs(PLOT_DIR, exist_ok=True)

    feat_cols = [c for c in ALL_FEATURE_COLS if c in full_df.columns]
    feat_cols_full = feat_cols + ["horizon", "chronos_pred"]

    item_ids = sorted(full_df["item_id"].unique())
    summary_rows = []

    print(f"\n[Evaluate] {len(item_ids)} items ...", flush=True)
    for item_id in tqdm(item_ids, desc="items"):
        df_item = full_df[full_df["item_id"] == item_id].sort_values("date").reset_index(drop=True)
        n = len(df_item)

        if n < ctx + pred_len:
            print(f"  [{item_id}] 데이터 부족 ({n}행), SKIP", flush=True)
            continue

        target_arr = df_item["target"].values.astype(float)
        context_arr = target_arr[: n - pred_len][-ctx:]
        test_target = target_arr[n - pred_len :]
        test_df_slice = df_item.iloc[n - pred_len :].reset_index(drop=True)

        # Chronos-2 예측 (full samples for band)
        base_median, base_samples = _chronos_forecast_full(pipeline, context_arr, pred_len)

        # LightGBM 잔차 보정
        X_lgb_rows = []
        for h in range(pred_len):
            row_feat = {}
            for col in feat_cols:
                row_feat[col] = test_df_slice[col].iat[h] if col in test_df_slice.columns else np.nan
            row_feat["horizon"] = h + 1
            row_feat["chronos_pred"] = base_median[h]
            X_lgb_rows.append(row_feat)
        X_lgb = pd.DataFrame(X_lgb_rows)[feat_cols_full]
        residual_pred = booster.predict(X_lgb)

        boosted_median = base_median + residual_pred
        boosted_samples = base_samples + residual_pred[np.newaxis, :]

        # arcsinh → 실제 가격
        actual_real      = np.sinh(test_target)
        base_real        = np.sinh(base_median)
        boosted_real     = np.sinh(boosted_median)
        boosted_q10      = np.sinh(np.percentile(boosted_samples, 10, axis=0))
        boosted_q90      = np.sinh(np.percentile(boosted_samples, 90, axis=0))

        past_real = np.sinh(context_arr[-past_n:])

        def _m(y, yhat, label):
            r = math.sqrt(mean_squared_error(y, yhat))
            m = mean_absolute_error(y, yhat)
            p = mape(y, yhat)
            return {"item_id": item_id, "method": label, "RMSE": round(r, 1),
                    "MAE": round(m, 1), "MAPE(%)": round(p, 1)}

        m_base = _m(actual_real, base_real, "chronos_base")
        m_boost = _m(actual_real, boosted_real, "residual_boost")
        summary_rows.append(m_base)
        summary_rows.append(m_boost)

        # ── TimesFM 스타일 시각화 ──
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        past_x   = np.arange(-past_n, 0)
        future_x = np.arange(0, pred_len)

        for ax, title, pred_r, q10, q90, metrics, color in [
            (axes[0], "Chronos-2 Zero-Shot (Base)",
             base_real, np.sinh(np.percentile(base_samples, 10, axis=0)),
             np.sinh(np.percentile(base_samples, 90, axis=0)), m_base, "red"),
            (axes[1], "Chronos-2 + Residual Boost (LightGBM)",
             boosted_real, boosted_q10, boosted_q90, m_boost, "red"),
        ]:
            ax.plot(past_x, past_real, color="gray", linewidth=1.2,
                    label=f"Past ({past_n}d actual)")
            ax.plot(future_x, actual_real, color="green", marker="o",
                    markersize=4, linewidth=1.5, label="Future (actual)")
            ax.plot(future_x, pred_r, color=color, linestyle="--",
                    marker="x", markersize=5, linewidth=1.5, label="Forecast (predicted)")
            ax.fill_between(future_x, q10, q90, color=color, alpha=0.12,
                            label="Prediction Band (10~90%)")
            ax.axvline(0, color="black", linestyle=":", linewidth=0.8)

            ax.set_title(f"{title}: {item_id}", fontsize=13, fontweight="bold")
            ax.set_ylabel("Price (KRW)")
            ax.legend(loc="upper left", fontsize=8)
            ax.grid(True, alpha=0.25)

            stats_txt = (f"RMSE={metrics['RMSE']:.1f}  "
                         f"MAE={metrics['MAE']:.1f}  "
                         f"MAPE={metrics['MAPE(%)']:.1f}%")
            ax.text(0.02, 0.96, stats_txt, transform=ax.transAxes, fontsize=9,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                              edgecolor="gray", alpha=0.85))

        axes[1].set_xlabel("Days (0 = forecast origin)")
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, f"residual_boost_{item_id}.png"), dpi=150)
        plt.close()

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(OUTPUT_DIR, "residual_boost_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n[Summary] {summary_path}")
    print(summary_df.to_string(index=False))

    # 개선율 요약
    base_df  = summary_df[summary_df["method"] == "chronos_base"].set_index("item_id")
    boost_df = summary_df[summary_df["method"] == "residual_boost"].set_index("item_id")
    improve = pd.DataFrame({
        "Base_MAPE":  base_df["MAPE(%)"],
        "Boost_MAPE": boost_df["MAPE(%)"],
    })
    improve["Improvement(%)"] = ((improve["Base_MAPE"] - improve["Boost_MAPE"])
                                  / improve["Base_MAPE"] * 100).round(1)
    improve_path = os.path.join(OUTPUT_DIR, "improvement_summary.csv")
    improve.to_csv(improve_path)
    print(f"\n[Improvement] {improve_path}")
    print(improve.to_string())

    return summary_df


# ==========================================
# main
# ==========================================
def main():
    print("=" * 60)
    print(" Residual Boosting: Chronos-2 + LightGBM  (ALL items)")
    print("=" * 60)

    # 전체 데이터 로드 (train + test 합치기)
    train_df = pd.read_csv(CONFIG["train_path"], parse_dates=["date"])
    test_df  = pd.read_csv(CONFIG["test_path"],  parse_dates=["date"])
    full_df  = pd.concat([train_df, test_df], ignore_index=True)
    full_df  = full_df.sort_values(["item_id", "date"]).reset_index(drop=True)

    all_items = sorted(full_df["item_id"].unique())
    pred_len  = CONFIG["prediction_length"]
    ctx       = CONFIG["context_length"]

    # train = 각 item에서 마지막 14일 제외
    train_parts = []
    for item_id in all_items:
        df_item = full_df[full_df["item_id"] == item_id].sort_values("date")
        if len(df_item) >= ctx + pred_len:
            train_parts.append(df_item.iloc[:-pred_len])
    train_for_residual = pd.concat(train_parts, ignore_index=True)

    print(f"Total items: {len(all_items)}")
    print(f"Train rows (residual): {len(train_for_residual):,}")

    print("\n[1/4] Loading Chronos-2 ...", flush=True)
    pipeline = TSFMPipeline.from_pretrained(
        CONFIG["model_id"], device_map="auto", torch_dtype=torch.float16,
    )
    pipeline.model.eval()

    print("\n[2/4] Collecting train residuals (rolling) ...", flush=True)
    eligible_items = [it for it in all_items
                      if len(full_df[full_df["item_id"] == it]) >= ctx + pred_len]
    residual_df = collect_train_residuals(pipeline, train_for_residual, eligible_items)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    residual_df.to_csv(os.path.join(OUTPUT_DIR, "train_residuals.csv"), index=False)
    print(f"  Residual rows: {len(residual_df):,}")

    print("\n[3/4] Training LightGBM residual model ...", flush=True)
    booster = train_residual_model(residual_df)
    booster.save_model(os.path.join(OUTPUT_DIR, "residual_lgb_model.txt"))

    print("\n[4/4] Evaluating ALL items ...", flush=True)
    evaluate_all_items(pipeline, booster, full_df)

    print("\n[DONE]")


if __name__ == "__main__":
    main()
