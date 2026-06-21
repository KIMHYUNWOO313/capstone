from __future__ import annotations

import threading
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parents[1]
POINT_DIR = BASE_DIR / "POINT_FORECAST"
PROB_DIR = BASE_DIR / "PROBABILISTIC_FORECAST"

POINT_DATA_PATH = POINT_DIR / "data" / "full_baseline.parquet"
PROB_DATA_PATH = PROB_DIR / "data" / "full_baseline.parquet"
PROB_META_PATH = PROB_DIR / "data" / "meta_baseline.json"
PROB_MODEL_DIR = PROB_DIR / "model"
WEATHER_DIR = BASE_DIR / "WEATHER_FORECAST"

POINT_MODEL_ID = "google/timesfm-2.5-200m-pytorch"
POINT_CONTEXT_LEN = 384
POINT_HORIZON = 3
PROB_MODEL_NAME = "Chronos2LoRA_baseline"


class ForecastRequest(BaseModel):
    model: Literal["point", "probabilistic", "both"] = Field(default="both")
    item_id: str | None = Field(
        default=None,
        description="예측할 item_id. 비우면 전체 46개 품목을 예측합니다.",
    )


class PointForecaster:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model = None
        self._df: pd.DataFrame | None = None

    def _load_data(self) -> pd.DataFrame:
        if self._df is None:
            df = pd.read_parquet(POINT_DATA_PATH)
            if isinstance(df.index, pd.MultiIndex):
                df = df.reset_index()
            self._df = df.sort_values(["item_id", "timestamp"]).reset_index(drop=True)
        return self._df

    def _load_model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    import timesfm

                    if hasattr(timesfm, "TimesFM_2p5_200M_torch"):
                        from timesfm.configs import ForecastConfig

                        # from_pretrained() can pass hub-mixin-only kwargs that
                        # current TimesFM does not accept with huggingface-hub<1.
                        model = timesfm.TimesFM_2p5_200M_torch._from_pretrained(
                            model_id=POINT_MODEL_ID,
                            revision=None,
                            cache_dir=None,
                            force_download=False,
                            local_files_only=False,
                            token=None,
                        )
                        model.compile(
                            ForecastConfig(
                                max_context=POINT_CONTEXT_LEN,
                                max_horizon=32,
                                normalize_inputs=True,
                                return_backcast=False,
                            )
                        )
                        self._model = ("timesfm_25", model)
                    else:
                        hparams = timesfm.TimesFmHparams(
                            context_len=POINT_CONTEXT_LEN,
                            horizon_len=32,
                            input_patch_len=32,
                            output_patch_len=128,
                            backend="cpu",
                            point_forecast_mode="mean",
                        )
                        checkpoint = timesfm.TimesFmCheckpoint(
                            version="pytorch",
                            huggingface_repo_id=POINT_MODEL_ID,
                        )
                        self._model = ("timesfm_legacy", timesfm.TimesFm(hparams, checkpoint))
        return self._model

    def _forecast(self, inputs: list[np.ndarray]) -> np.ndarray:
        model_kind, model = self._load_model()
        if model_kind == "timesfm_25":
            point_fc, _ = model.forecast(horizon=POINT_HORIZON, inputs=inputs)
        else:
            point_fc, _ = model.forecast(inputs=inputs, freq=[0] * len(inputs), normalize=True)
        return point_fc[:, :POINT_HORIZON]

    def items(self) -> list[str]:
        df = self._load_data()
        return sorted(df["item_id"].unique().tolist())

    def predict(self, item_id: str | None) -> dict:
        df = self._load_data()
        if item_id:
            if item_id not in set(df["item_id"].unique()):
                raise HTTPException(status_code=404, detail=f"Unknown item_id: {item_id}")
            grouped = [(item_id, df[df["item_id"] == item_id])]
        else:
            grouped = list(df.groupby("item_id"))

        items: list[str] = []
        inputs: list[np.ndarray] = []
        last_timestamps: dict[str, pd.Timestamp] = {}

        for iid, sub in grouped:
            sub = sub.sort_values("timestamp")
            arr = sub["target"].to_numpy(dtype=np.float32)
            if len(arr) < 64:
                continue
            items.append(iid)
            inputs.append(arr[-POINT_CONTEXT_LEN:])
            last_timestamps[iid] = pd.Timestamp(sub["timestamp"].max())

        if not inputs:
            raise HTTPException(status_code=422, detail="예측 가능한 시계열이 없습니다.")

        point_fc = self._forecast(inputs)

        rows = []
        for iid, preds in zip(items, point_fc):
            last_ts = last_timestamps[iid]
            for step, value in enumerate(preds[:POINT_HORIZON], start=1):
                rows.append(
                    {
                        "model": "POINT_FORECAST",
                        "model_detail": "점예측(3일)",
                        "item_id": iid,
                        "timestamp": (last_ts + pd.Timedelta(days=step)).date().isoformat(),
                        "horizon_step": step,
                        "y_pred": float(value),
                    }
                )

        return {
            "description": "점예측(3일) 기반 가격 예측",
            "rows": rows,
        }


class ProbabilisticForecaster:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._predictor = None
        self._full = None
        self._full_reset: pd.DataFrame | None = None
        self._known_covs: list[str] | None = None
        self._item_station_map: dict[str, str] | None = None

    def _load_runtime(self):
        if self._predictor is None:
            with self._lock:
                if self._predictor is None:
                    import json
                    import pathlib
                    from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

                    # The bundled AutoGluon predictor was packaged on Windows.
                    # This lets pathlib paths inside pickles load on Linux.
                    pathlib.WindowsPath = pathlib.PosixPath  # type: ignore[misc, assignment]
                    predictor = TimeSeriesPredictor.load(PROB_MODEL_DIR)
                    with PROB_META_PATH.open(encoding="utf-8") as f:
                        meta = json.load(f)
                    full = TimeSeriesDataFrame(pd.read_parquet(PROB_DATA_PATH))
                    self._predictor = predictor
                    self._full = full
                    self._full_reset = full.reset_index()
                    self._known_covs = list(meta["known_covariates"])
                    self._item_station_map = dict(meta.get("item_station_map") or {})
        return self._predictor, self._full, self._full_reset, self._known_covs, self._item_station_map

    def items(self) -> list[str]:
        _, _, full_reset, _, _ = self._load_runtime()
        assert full_reset is not None
        return sorted(full_reset["item_id"].unique().tolist())

    def _make_known(self, context):
        from autogluon.timeseries import TimeSeriesDataFrame

        predictor, _, full_reset, known_covs, item_station_map = self._load_runtime()
        assert full_reset is not None
        assert known_covs is not None

        if WEATHER_DIR.is_dir() and item_station_map:
            try:
                if str(WEATHER_DIR) not in sys.path:
                    sys.path.insert(0, str(WEATHER_DIR))
                from covariate_builder import KNOWN_COVS as WEATHER_KNOWN_COVS
                from covariate_builder import build_known_covariates_frame

                context_items = set(context.reset_index()["item_id"].astype(str).unique())
                station_map = {
                    item_id: station
                    for item_id, station in item_station_map.items()
                    if item_id in context_items
                }
                known_df = build_known_covariates_frame(
                    full_reset,
                    station_map,
                    horizon=getattr(predictor, "prediction_length", 10),
                )
                if not known_df.empty:
                    known_df = known_df[["item_id", "timestamp", *WEATHER_KNOWN_COVS]]
                    return TimeSeriesDataFrame.from_data_frame(
                        known_df,
                        id_column="item_id",
                        timestamp_column="timestamp",
                    )
            except Exception as exc:
                print(f"[known_covariates] weather forecast mode failed, fallback to lookup: {exc!r}")

        fd = predictor.make_future_data_frame(context).reset_index()
        fd = fd.merge(
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
        fd = fd.merge(last_known, on="item_id", how="left")

        for col in known_covs:
            fd[col] = fd[col].fillna(fd[f"{col}__last"]).fillna(0)
            fd = fd.drop(columns=[f"{col}__last"])

        return TimeSeriesDataFrame.from_data_frame(
            fd,
            id_column="item_id",
            timestamp_column="timestamp",
        )

    def predict(self, item_id: str | None) -> dict:
        predictor, full, full_reset, _, _ = self._load_runtime()
        assert full is not None
        assert full_reset is not None

        if item_id:
            if item_id not in set(full_reset["item_id"].unique()):
                raise HTTPException(status_code=404, detail=f"Unknown item_id: {item_id}")
            context = full.loc[[item_id]]
        else:
            context = full

        known_fc = self._make_known(context)
        forecast = predictor.predict(context, known_covariates=known_fc, model=PROB_MODEL_NAME)
        fc = forecast.reset_index()

        rows = []
        for record in fc.to_dict(orient="records"):
            row = {
                "model": "PROBABILISTIC_FORECAST",
                "model_detail": "구간 예측(10일)",
                "item_id": record["item_id"],
                "timestamp": pd.Timestamp(record["timestamp"]).date().isoformat(),
                "mean": _float_or_none(record.get("mean")),
                "p10": _float_or_none(record.get("0.1")),
                "p20": _float_or_none(record.get("0.2")),
                "p50": _float_or_none(record.get("0.5")),
                "p80": _float_or_none(record.get("0.8")),
                "p90": _float_or_none(record.get("0.9")),
            }
            rows.append(row)

        return {
            "description": "구간 예측(10일) 기반 확률범위 예측(9분위 + mean)",
            "rows": rows,
        }


def _float_or_none(value) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


point_forecaster = PointForecaster()
prob_forecaster = ProbabilisticForecaster()

app = FastAPI(title="Agricultural Price Forecast Web", version="1.0.0")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    template = Path(__file__).with_name("templates") / "index.html"
    return template.read_text(encoding="utf-8")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "time": datetime.now(timezone.utc).isoformat(),
        "models": {
            "point": "점예측(3일)",
            "probabilistic": "구간 예측(10일)",
        },
    }


@app.get("/api/items")
def list_items() -> dict:
    # POINT 데이터와 PROB 데이터는 동일한 46개 품목을 대상으로 합니다.
    return {"items": point_forecaster.items()}


@app.post("/api/predict")
def predict(req: ForecastRequest) -> dict:
    result = {
        "requested_model": req.model,
        "item_id": req.item_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "outputs": {},
    }

    if req.model in {"point", "both"}:
        result["outputs"]["point"] = point_forecaster.predict(req.item_id)

    if req.model in {"probabilistic", "both"}:
        result["outputs"]["probabilistic"] = prob_forecaster.predict(req.item_id)

    return result
