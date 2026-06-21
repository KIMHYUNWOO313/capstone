"""xai_explainer 공통 설정."""
import os
from pathlib import Path

from dotenv import load_dotenv

PKG_DIR = Path(__file__).resolve().parent
INPUT_DIR = PKG_DIR / "inputs"
OUTPUT_DIR = PKG_DIR / "outputs"

load_dotenv(PKG_DIR / ".env")

GPT_API_KEY = os.getenv("GPT_API_KEY", "").strip()
WARN_API_KEY = os.getenv("WARN_API_KEY", "").strip()

GPT_MODEL = os.getenv("GPT_MODEL", "gpt-4o")

FORECAST_10D_PATH = INPUT_DIR / "forecast_10day.parquet"
FORECAST_3D_PATH = INPUT_DIR / "forecast_3day.parquet"
FULL_BASELINE_PATH = INPUT_DIR / "full_baseline.parquet"
META_PATH = INPUT_DIR / "meta_baseline.json"
DATASET_DESC_DIR = INPUT_DIR / "dataset_descriptions"
AGRI_REPORT_DIR = INPUT_DIR / "agri_report"

WARN_API_BASE = "http://apis.data.go.kr/1360000/WthrWrnInfoService"


def require_keys() -> None:
    """필수 API 키 누락 시 즉시 실패."""
    missing = []
    if not GPT_API_KEY:
        missing.append("GPT_API_KEY")
    if not WARN_API_KEY:
        missing.append("WARN_API_KEY")
    if missing:
        raise RuntimeError(
            f".env에 다음 키가 비어 있습니다: {', '.join(missing)}. "
            f".env.example을 참고해 채워주세요."
        )
