"""사전 캐싱된 두 모델의 예측 결과 로드."""
import pandas as pd

from config import FORECAST_10D_PATH, FORECAST_3D_PATH


def _load(path, label):
    if not path.exists():
        raise FileNotFoundError(
            f"{label} 예측 결과 파일을 찾을 수 없습니다: {path}\n"
            f"scripts/refresh_forecasts.py 를 실행하여 inputs/ 에 생성하세요."
        )
    df = pd.read_parquet(path)
    if "item_id" not in df.columns and df.index.name != "item_id":
        df = df.reset_index()
    return df


def load_forecasts() -> dict[str, pd.DataFrame]:
    """
    Returns:
        {
          "chronos2": DataFrame(item_id, timestamp, 0.1..0.9, mean),
          "timesfm":  DataFrame(item_id, timestamp, y_pred),
        }
    """
    return {
        "chronos2": _load(FORECAST_10D_PATH, "Chronos2 10일 확률예측"),
        "timesfm":  _load(FORECAST_3D_PATH, "TimesFM 3일 점예측"),
    }


def slice_item(forecasts: dict[str, pd.DataFrame], item_id: str) -> dict[str, pd.DataFrame]:
    out = {}
    for k, df in forecasts.items():
        sub = df[df["item_id"] == item_id].sort_values("timestamp").reset_index(drop=True)
        out[k] = sub
    return out


if __name__ == "__main__":
    fc = load_forecasts()
    for k, df in fc.items():
        print(f"\n=== {k} ===")
        print(df.head(5).to_string(index=False))
        print(f"items: {df['item_id'].nunique()}  rows: {len(df)}")
