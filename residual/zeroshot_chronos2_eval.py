"""
zeroshot_chronos2_eval.py
=========================
Chronos-2 베이스 모델을 이용한 제로샷(Zero-shot) 추론 평가 스크립트.
test_data.csv를 읽어 작물당 1회 14일 예측을 수행하고 결과를 시각화합니다.
"""

import os
import math
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from gluonts.dataset.common import ListDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error

from chronos.chronos2.pipeline import Chronos2Pipeline as TSFMPipeline

# ==========================================
# CONFIGURATION
# ==========================================
CONFIG = {
    "model_id":         "amazon/chronos-2",
    "test_data_path":   "data/production_split/test_data.csv",
    "context_length":   365,
    "prediction_length": 14,
}

OUTPUT_DIR = "ai/zeroshot_output"
PLOT_DIR   = f"{OUTPUT_DIR}/plots"

# Covariate 컬럼 정의 (데이터 딕셔너리 기준)
FUTURE_COVARIATE_COLS = [
    'month', 'temp_avg', 'rain_sum', 'humid_avg', 'sunshine_sum',
    'temp_diff', 'dayofweek', 'weekofyear', 'month_sin', 'month_cos',
    'dow_sin', 'dow_cos', 'weather_index', 'rain_impact'
]

PAST_COVARIATE_COLS = [
    'oil_diesel', 'cpi_total', 'gov_bond_3y', 'epu', 'm2_sa',
    'price_lag_1', 'price_lag_3', 'price_lag_7', 'price_lag_14',
    'price_lag_28', 'price_diff', 'price_ma7', 'temp_rolling_mean_7',
    'oil_diesel_lag_1', 'oil_diesel_lag_3', 'temp_avg_lag_1', 'rain_sum_lag_1'
]

# ==========================================
# 1. 데이터 로드
# ==========================================
def load_and_prepare_dataset(csv_path: str) -> list[dict]:
    print(f"\n[Loading Data] {csv_path}")
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df.sort_values(["item_id", "date"]).reset_index(drop=True)

    dataset_list = []
    for item_id, group in tqdm(df.groupby("item_id")):
        group = group.sort_values("date").reset_index(drop=True)
        target     = group["target"].values
        start_date = group["date"].iloc[0]

        entry: dict = {"target": target, "start": start_date, "item_id": item_id}

        if all(c in group.columns for c in FUTURE_COVARIATE_COLS):
            entry["feat_dynamic_real"] = group[FUTURE_COVARIATE_COLS].values.T
        if all(c in group.columns for c in PAST_COVARIATE_COLS):
            entry["past_feat_dynamic_real"] = group[PAST_COVARIATE_COLS].values.T

        dataset_list.append(entry)

    return dataset_list

# ==========================================
# 2. 평가 지표 및 예측
# ==========================================
def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    mask = actual != 0
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100

def run_zeroshot_evaluation(pipeline, test_raw_list: list[dict]):
    context_length   = CONFIG["context_length"]
    prediction_length = CONFIG["prediction_length"]

    os.makedirs(PLOT_DIR, exist_ok=True)
    summary_rows = []

    print("\n[Zero-shot] Inference started...")
    for ts_data in tqdm(test_raw_list):
        item_id     = ts_data["item_id"]
        full_target = ts_data["target"]

        if len(full_target) < context_length + prediction_length:
            print(f"  ⚠ [{item_id}] 데이터 부족, SKIP.")
            continue

        # Context / Ground-Truth 분리
        context_arcsinh = full_target[:context_length]
        actuals_arcsinh = full_target[context_length:]

        # Input 구성
        pred_input = {"target": context_arcsinh}
        past_cov_dict   = {}
        future_cov_dict = {}

        if "feat_dynamic_real" in ts_data:
            full_fdc = ts_data["feat_dynamic_real"]
            for i in range(full_fdc.shape[0]):
                key = f"future_known_cov_{i}"
                past_cov_dict[key]   = full_fdc[i, :context_length]
                future_cov_dict[key] = full_fdc[i, context_length:]

        if "past_feat_dynamic_real" in ts_data:
            past_arr = ts_data["past_feat_dynamic_real"]
            for i in range(past_arr.shape[0]):
                past_cov_dict[f"past_only_cov_{i}"] = past_arr[i, :context_length]

        if past_cov_dict:   pred_input["past_covariates"] = past_cov_dict
        if future_cov_dict: pred_input["future_covariates"] = future_cov_dict

        # 추론
        forecast_result = pipeline.predict([pred_input], prediction_length=prediction_length)
        forecast_obj    = list(forecast_result)[0]
        forecast_arcsinh = np.median(forecast_obj.numpy()[0], axis=0)

        # Arcsinh 역변환
        actuals_real  = np.sinh(actuals_arcsinh)
        forecast_real = np.sinh(forecast_arcsinh)

        # 지표 계산
        rmse = math.sqrt(mean_squared_error(actuals_real, forecast_real))
        mae_val  = mean_absolute_error(actuals_real, forecast_real)
        mape_val = mape(actuals_real, forecast_real)

        summary_rows.append({"item_id": item_id, "RMSE": rmse, "MAE": mae_val, "MAPE(%)": mape_val})

        # 그래프 생성
        past_30_arcsinh = full_target[context_length - 30: context_length]
        past_30_real    = np.sinh(past_30_arcsinh)

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.set_title(f"Zero-shot Forecast: {item_id}", fontsize=13, fontweight="bold")
        past_x    = range(-30, 0)
        future_x  = range(0, prediction_length)

        ax.plot(past_x,   past_30_real,  color="gray",  label="Past (30d actual)")
        ax.plot(future_x, actuals_real,  color="green", label="Future (actual)", marker="o", markersize=4)
        ax.plot(future_x, forecast_real, color="red",   label="Forecast (predicted)", linestyle="--", marker="x", markersize=5)

        ax.axvline(0, color="black", linestyle=":")
        ax.set_xlabel("Days")
        ax.set_ylabel("Price (KRW)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        stats_txt = f"RMSE={rmse:.1f}  MAE={mae_val:.1f}  MAPE={mape_val:.1f}%"
        ax.text(0.02, 0.96, stats_txt, transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.7))

        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f"zeroshot_{item_id}.png"), dpi=150)
        plt.close()

    # 요약 저장
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(OUTPUT_DIR, "zeroshot_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n[DONE] Summary saved: {summary_path}")
    print(summary_df.to_string(index=False))

def main():
    print(f"\n[Zero-shot Evaluation] {CONFIG['model_id']}")
    
    test_raw_list = load_and_prepare_dataset(CONFIG["test_data_path"])
    
    pipeline = TSFMPipeline.from_pretrained(
        CONFIG["model_id"],
        device_map="auto",
        torch_dtype=torch.float16,
    )
    pipeline.model.eval()

    run_zeroshot_evaluation(pipeline, test_raw_list)

if __name__ == "__main__":
    main()
