# BACKEND_HANDOVER — 백엔드 담당자 인계 매뉴얼

**작성일**: 2026-05-18
**대상**: 본 폴더 (`final_handoff/`) 를 인수받아 운영 환경에 통합할 백엔드 엔지니어
**소요 시간 가이드**: 환경 구축 1시간 + 첫 추론 30분 + 운영 통합 1~2일

본 매뉴얼은 백엔드 엔지니어가 **각 모델·데이터·API 의 작동 원리를 이해**하고, **운영 환경에 통합·운용**할 수 있도록 작성된 단계별 가이드입니다. 알고리즘 / ML 배경 지식이 깊지 않아도 따라할 수 있도록 작성되었습니다.

---

## 목차

1. [시스템 개요](#1-시스템-개요)
2. [폴더 구조와 각 모듈의 역할](#2-폴더-구조와-각-모듈의-역할)
3. [환경 설정 (Python · GPU · 패키지)](#3-환경-설정)
4. [API 키 관리](#4-api-키-관리)
5. [일일 운영 워크플로우](#5-일일-운영-워크플로우)
6. [각 모듈 사용법 — API · 입출력 스키마](#6-각-모듈-사용법--api--입출력-스키마)
7. [데이터 갱신 SOP](#7-데이터-갱신-sop)
8. [운영 시 주의사항](#8-운영-시-주의사항)
9. [트러블슈팅](#9-트러블슈팅)
10. [모니터링 권장](#10-모니터링-권장)
11. [부록: 핵심 데이터 스키마 한눈에](#11-부록-핵심-데이터-스키마)

---

## 1. 시스템 개요

### 1-1. 두 가지 예측 모델

| 모델 | 폴더 | 출력 | 호출 시 무엇이 필요한가 |
|---|---|---|---|
| **점예측** (3일) | `03_POINT_FORECAST` | `(item_id, timestamp, y_pred)` | 가격 시리즈만 (univariate) |
| **구간예측** (10일) | `02_PROBABILISTIC_FORECAST` | `(item_id, timestamp, 0.1~0.9, mean)` | 가격 + known covariate (일기예보 포함) |

### 1-2. 데이터 + 보조 모듈

| 폴더 | 역할 |
|---|---|
| `01_DATA/` | 학습/추론용 통합 데이터 (parquet) + 메타 정보 |
| `05_WEATHER_FORECAST/` | 기상청 일기예보 → known_covariates 빌더 (운영용) |
| `04_XAI_EXPLAINER/` | GPT-4o 기반 예측 사후 설명 모듈 (선택) |
| `99_REFERENCES/` | 기상청 API 활용가이드 등 외부 문서 |

### 1-3. 두 모델의 관계

```
                  ┌────────────────────────────┐
                  │ 01_DATA/data/full_baseline │   매일 갱신 (today-1까지)
                  └─────────────┬──────────────┘
                                │
        ┌───────────────────────┼────────────────────────┐
        │                       │                        │
        ▼                       ▼                        ▼
   ┌─────────┐             ┌─────────┐              ┌─────────┐
   │ 03 점예측 │             │ 05 일기예보│              │ 02 구간예측│
   │TimesFM ZS│             │ Weather  │─known────▶│Chronos2  │
   │univariate│             │   API    │ covariates │  LoRA    │
   └────┬─────┘             └─────────┘             └────┬─────┘
        │                                                 │
        ▼                                                 ▼
   y_pred (3일)                                  9분위 + mean (10일)
```

- **점예측**은 가격만 보고 예측 → **외부 의존 적음, 가장 robust**
- **구간예측**은 일기예보 + 그 외 known을 받음 → 운영 시 `05_WEATHER_FORECAST` 의존

---

## 2. 폴더 구조와 각 모듈의 역할

```
final_handoff/
├── 00_README.md                ← 전체 인덱스 (먼저 읽기)
├── 00_INTEGRATED_REPORT.md     ← 통합 narrative 보고서
├── EXPERIMENT_REPORT.md        ← 정량 실험 결과 (발표용)
├── BACKEND_HANDOVER.md         ← 본 문서
├── VERIFICATION.md             ← 실행 검증 로그
│
├── 01_DATA/                                  ← 데이터·전처리·variant 비교
│   ├── README.md
│   ├── dataset_description_probabilistic.md   ← 변수 사전 (확률예측 관점)
│   ├── dataset_description_point.md           ← 변수 사전 (점예측 관점)
│   ├── meta_baseline.json                     ← known/past covariate 분류 + item_station_map
│   └── data/
│       ├── full_baseline.parquet                       (172,868 × 13)
│       ├── full_baseline_extended_20260516.parquet     ★ 운영 시 사용
│       └── static_baseline.parquet                     (46 × 4, 정적 속성)
│
├── 02_PROBABILISTIC_FORECAST/   ← 모델 1: Chronos2 LoRA (10일)
│   ├── README.md
│   ├── predict_example.py                     ← 추론 진입점 (forecast/lookup 모드)
│   ├── requirements.txt
│   ├── experiments/                           ← 6-way 비교 + 차트
│   ├── sample_forecasts/                      ← 13 작물 fan chart
│   └── model/                                  ← AutoGluon predictor (LoRA adapter 포함)
│
├── 03_POINT_FORECAST/           ← 모델 2: TimesFM 2.5 ZS (3일)
│   ├── README.md
│   ├── predict_example.py                     ← 추론 진입점 (단변량 ZS)
│   ├── requirements.txt
│   ├── experiments/                           ← 7-way ablation + 차트
│   └── sample_forecasts/                      ← 3-모델 비교
│
├── 04_XAI_EXPLAINER/            ← 선택: GPT-4o 사후 설명 모듈
│   ├── README.md
│   ├── run_xai.py                             ← XAI 진입점
│   ├── explain.py / config.py / prompt_builder.py
│   ├── data_loaders/
│   ├── scripts/refresh_forecasts.py           ← 두 모델 forecast 캐시 갱신
│   ├── inputs/                                ← 예측 캐시 + 농업월보 PDF
│   ├── outputs/                               ← 실행 결과 JSON
│   └── .env.example
│
├── 05_WEATHER_FORECAST/         ← 운영 필수: 기상청 일기예보 모듈
│   ├── README.md
│   ├── stn_grid_map.py                        ← 산지 18개 → (nx,ny) + regId
│   ├── weather_fetcher.py                     ← 단기/중기 API 호출
│   ├── covariate_builder.py                   ← known_covariates 빌드 (운영 진입점)
│   ├── requirements.txt
│   └── .env / .env.example                    ← API 키
│
└── 99_REFERENCES/               ← 기상청 API 활용가이드 등
```

---

## 3. 환경 설정

### 3-1. Python 버전

- **권장**: Python 3.10 또는 3.11
- 3.12+ 는 timesfm 호환성 미검증

### 3-2. GPU 권장사항

| 모듈 | GPU 요구 | 메모리 |
|---|---|---|
| 02_PROBABILISTIC (Chronos2 LoRA) | 강력 권장 (CPU 5~10배 느림) | 4~6 GB |
| 03_POINT (TimesFM ZS) | 권장 | 3~5 GB |
| 04_XAI (GPT-4o 호출) | 불필요 | - |
| 05_WEATHER (API 호출) | 불필요 | - |

**검증된 GPU**: NVIDIA RTX 5060 Ti (16 GB), CUDA 12.x

### 3-3. 환경 분리 — autogluon ↔ timesfm 충돌 회피

`autogluon.timeseries==1.5.0` 은 `transformers<4.58` 만 호환. TimesFM 의 PEFT LoRA 학습을 다시 시도하려면 `transformers 5.x` 필요. **두 환경 분리 권장**.

```bash
# 환경 1: 확률예측 (Chronos2)
conda create -n prob_forecast python=3.11
conda activate prob_forecast
cd 02_PROBABILISTIC_FORECAST
pip install -r requirements.txt          # autogluon.timeseries==1.5.0
cd ../05_WEATHER_FORECAST
pip install -r requirements.txt          # requests, python-dotenv

# 환경 2: 점예측 (TimesFM)
conda create -n point_forecast python=3.11
conda activate point_forecast
cd 03_POINT_FORECAST
pip install -r requirements.txt          # timesfm>=2.0.0

# 환경 3: XAI (선택)
conda create -n xai python=3.11
conda activate xai
cd 04_XAI_EXPLAINER
pip install -r requirements.txt          # openai, python-dotenv, requests, pypdf
```

대안: 단일 환경 (anaconda capstone) 에서 모두 통합 운영도 가능. 단 `PYTHONNOUSERSITE=1 PYTHONIOENCODING=utf-8 PYTHONUTF8=1` 환경변수 권장 (Windows + bash 한글 호환).

### 3-4. 모델 가중치 다운로드

| 모델 | 위치 | 크기 | 비고 |
|---|---|---|---|
| Chronos2 LoRA | `02_PROBABILISTIC_FORECAST/model/` | ~10 MB | 폴더에 동봉 (즉시 로드) |
| TimesFM 2.5 | HuggingFace `google/timesfm-2.5-200m-pytorch` | ~200 MB | 첫 실행 시 자동 다운로드 |

오프라인 환경: `~/.cache/huggingface/hub/` 에 TimesFM 캐시 사전 복사.

---

## 4. API 키 관리

### 4-1. 필요한 외부 API 키 4종

| API | 용도 | 발급처 | 환경변수명 |
|---|---|---|---|
| 단기예보 (`getVilageFcst`) | Chronos2 known (D+1~D+3 일교차) | https://www.data.go.kr/data/15084084/openapi.do | `short_forecast_api` |
| 중기예보 (`getMidTa`) | Chronos2 known (D+4~D+10 일교차) | https://www.data.go.kr/data/15059468/openapi.do | `middle_forecast_api` |
| 기상특보 (`WthrWrnInfoService`) | XAI 모듈 (1주일치 특보) | https://www.data.go.kr/data/15059468/openapi.do | `WARN_API_KEY` |
| OpenAI GPT-4o | XAI 모듈 (사후 설명) | https://platform.openai.com/api-keys | `GPT_API_KEY` |

### 4-2. .env 파일 위치

| 파일 | 키 |
|---|---|
| `05_WEATHER_FORECAST/.env` | `short_forecast_api`, `middle_forecast_api` |
| `04_XAI_EXPLAINER/.env` | `GPT_API_KEY`, `WARN_API_KEY`, (옵션) `GPT_MODEL` |

### 4-3. 키 발급 절차

#### 기상청 키 (data.go.kr)

1. https://www.data.go.kr 회원가입 (개인 무료)
2. 위 단기/중기/특보 페이지에서 **활용신청** 클릭
3. 사용 목적 입력 → 자동 승인 (~1분)
4. 마이페이지 > 데이터활용 > 일반인증키 복사
5. `.env` 에 `short_forecast_api = <키>` 형식으로 저장
6. **3개 API 모두 같은 인증키** 가 발급되는 경우가 일반적 (한 계정 = 한 키)

#### OpenAI 키

1. https://platform.openai.com 회원가입
2. 결제 정보 등록 + 소액 충전 ($5~ 권장)
3. API Keys 메뉴 > Create new secret key
4. `04_XAI_EXPLAINER/.env` 에 `GPT_API_KEY = sk-...` 저장

### 4-4. 보안 가이드

- `.env` 는 **절대 git 커밋 금지**. 본 폴더의 `.env` 는 인계 편의를 위해 포함되어 있으나, 운영 환경 배포 전 별도 교체 권장
- 운영 환경에서는 **OS 환경변수 또는 비밀 관리 시스템 (AWS Secrets Manager / GCP Secret Manager / Vault)** 으로 주입
- 키 노출 의심 시 즉시 발급처에서 재발급

### 4-5. Rate Limit

| API | 제한 | 본 시스템 호출량 |
|---|---|---|
| 기상청 단기예보 | 30 tps, 일 1,000건 (기본) | 1일 ~18회 (산지당) |
| 기상청 중기예보 | 30 tps, 일 1,000건 | 1일 ~18회 |
| OpenAI gpt-4o | 분당 토큰 limit (계정별 다름) | 1일 ~46회 (품목당 1회) |

한 번의 known_covariates 빌드 = 단기 18회 + 중기 18회 = **최대 36회**. 일일 호출량은 한국식 발표시각 격차 4회 + 운영 호출 1~2회 = **~150회/일**. 기본 한도 내에서 충분.

---

## 5. 일일 운영 워크플로우

### 5-1. 권장 일정

```
06:10 — 기상청 06시 중기예보 발효 → 데이터 갱신 + 모델 추론
        (전날 도매가격 통계 + 오늘 새벽 기상예보)

06:30 — Chronos2 LoRA 10일 구간예측 + 일기예보 통합
06:35 — TimesFM 2.5 ZS 3일 점예측
06:40 — XAI 사후 설명 생성 (선택)
06:45 — 결과를 DB/캐시에 저장 → API 서버 반영

00:00~ — 사용자/대시보드에 노출
```

### 5-2. 매일 실행해야 할 스크립트

```bash
# 1. (사전) 01_DATA/data/full_baseline_extended_*.parquet 매일 갱신
#    → 도매가, 거래량, 기상, 거시경제, 에너지, 감성 5 도메인 어제분 추가
#    이 부분은 별도 ETL 파이프라인 (본 폴더 미포함)

# 2. Chronos2 LoRA 10일 구간예측 (forecast 모드, 일기예보 통합)
cd 02_PROBABILISTIC_FORECAST
python predict_example.py

# 3. TimesFM 2.5 3일 점예측 (변경 없음)
cd ../03_POINT_FORECAST
python predict_example.py

# 4. (선택) XAI 사후 설명 + 캐시 갱신
cd ../04_XAI_EXPLAINER
python scripts/refresh_forecasts.py    # forecast_10day.parquet, forecast_3day.parquet 갱신
python run_xai.py                       # 46품목 XAI 설명 → outputs/
```

### 5-3. 캐싱 권장

`05_WEATHER_FORECAST/covariate_builder.py` 는 **모듈 내 메모리 캐시** 만 가짐. 일일 한 번 호출이면 충분하므로 별도 디스크 캐시 불필요. 단, 빈번한 호출이 필요하면:

```python
# 예: scripts/cache_weather.py (백엔드가 직접 작성)
import json
from pathlib import Path
from covariate_builder import build_known_covariates_frame

# ... build known_df ...
cache_path = Path("cache/weather_known_YYYYMMDD.parquet")
known_df.to_parquet(cache_path)
```

---

## 6. 각 모듈 사용법 — API · 입출력 스키마

### 6-1. `02_PROBABILISTIC_FORECAST/predict_example.py`

#### CLI

```bash
python predict_example.py             # 기본: forecast 모드 (기상청 API)
python predict_example.py lookup       # 명시적 lookup 모드 (오프라인 평가용)
```

#### Python API 호출 패턴

```python
import sys
from pathlib import Path
import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

# 1. 모듈 import
HERE = Path("final_handoff")
sys.path.insert(0, str(HERE / "05_WEATHER_FORECAST"))
from covariate_builder import build_known_covariates_frame, KNOWN_COVS

# 2. 모델 + 데이터 로드
predictor = TimeSeriesPredictor.load(str(HERE / "02_PROBABILISTIC_FORECAST/model"))
import json
with open(HERE / "01_DATA/meta_baseline.json", encoding="utf-8") as f:
    meta = json.load(f)
item_station_map = meta["item_station_map"]

full_df = pd.read_parquet(HERE / "01_DATA/data/full_baseline_extended_20260516.parquet")
full = TimeSeriesDataFrame(full_df.set_index(["item_id", "timestamp"])
                          if "item_id" in full_df.columns else full_df)

# 3. known_covariates 빌드 (기상청 API)
known_df = build_known_covariates_frame(full_df, item_station_map, horizon=10)
known_df = known_df[["item_id", "timestamp"] + KNOWN_COVS]
known_fc = TimeSeriesDataFrame.from_data_frame(
    known_df, id_column="item_id", timestamp_column="timestamp"
)

# 4. 추론
forecast = predictor.predict(full, known_covariates=known_fc, model="Chronos2LoRA_baseline")
print(forecast.head())
```

#### 출력 스키마

| 컬럼 | 타입 | 의미 |
|---|---|---|
| item_id | str (index) | 품목 ID |
| timestamp | Timestamp (index) | 예측 일자 |
| 0.1, 0.2, ..., 0.9 | float | 9개 분위 예측값 |
| mean | float | 평균 예측 |

- **80% 신뢰구간**: `[0.1, 0.9]`
- **60% 신뢰구간**: `[0.2, 0.8]`
- **중앙값 (점예측 대용)**: `0.5`
- **행 수**: 46 품목 × 10 일 = **460 행**

### 6-2. `03_POINT_FORECAST/predict_example.py`

#### CLI

```bash
python predict_example.py     # 변경 없음 (univariate ZS)
```

#### Python API 호출 패턴

```python
import json
import numpy as np
import pandas as pd
import timesfm
from timesfm.configs import ForecastConfig

# 1. 모델 로드
model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
model.compile(ForecastConfig(
    max_context=384, max_horizon=32,
    normalize_inputs=True, return_backcast=False,
))

# 2. 데이터 로드
df = pd.read_parquet("01_DATA/data/full_baseline_extended_20260516.parquet")
if isinstance(df.index, pd.MultiIndex):
    df = df.reset_index()
df = df.sort_values(["item_id", "timestamp"]).reset_index(drop=True)

# 3. 추론 (target 시리즈만, univariate)
items, inputs = [], []
for iid, sub in df.groupby("item_id"):
    arr = sub.sort_values("timestamp")["target"].to_numpy(dtype=np.float32)[-384:]
    items.append(iid)
    inputs.append(arr)
point_fc, _ = model.forecast(horizon=3, inputs=inputs)
# point_fc.shape == (46, 3)

# 4. DataFrame 변환
rows = []
for iid, preds in zip(items, point_fc):
    last_ts = df[df["item_id"] == iid]["timestamp"].max()
    for k in range(3):
        rows.append({"item_id": iid, "timestamp": last_ts + pd.Timedelta(days=k+1), "y_pred": float(preds[k])})
out = pd.DataFrame(rows)
```

#### 출력 스키마

| 컬럼 | 타입 | 의미 |
|---|---|---|
| item_id | str | 품목 ID |
| timestamp | Timestamp | 예측 일자 |
| y_pred | float | 점예측 값 (원, 단위 raw) |

**행 수**: 46 품목 × 3 일 = **138 행**

### 6-3. `05_WEATHER_FORECAST/covariate_builder.py`

#### 주요 함수

```python
from covariate_builder import (
    build_known_covariates_frame,
    KNOWN_COVS,
    clear_caches,
)

# 빌드
known_df = build_known_covariates_frame(
    full_baseline=full_df,            # full_baseline parquet DataFrame
    item_station_map=meta["item_station_map"],  # dict[item_id, station_name]
    horizon=10,                       # 예측 horizon (Chronos2=10)
    today=None,                       # 기본 datetime.date.today()
)
# 반환: DataFrame(item_id, timestamp, market_rest, weather_temp_range,
#                weather_sunshine_dur, bok_base_rate, cpi_growth_rate)

# 일일 갱신 시 캐시 비우기
clear_caches()
```

#### 동작 흐름

1. 각 품목의 station 조회 (item_station_map)
2. 단기예보 (`getVilageFcst`) → D+1~D+3 일별 일교차
3. 중기예보 (`getMidTa`) → D+4~D+10 일교차
4. 나머지 known 4종 = 마지막 관측치 ffill
5. 캐시: 같은 (nx, ny) / regId 는 한 번만 호출

### 6-4. `04_XAI_EXPLAINER/run_xai.py`

```bash
python run_xai.py                                # 전체 46품목
python run_xai.py --items apple_fuji_box10kg_high  # 특정 품목
python run_xai.py --limit 3 --dry-run             # 3개만 dry-run
python run_xai.py --no-web-search                  # web_search 비활성화 (비용 절감)
python run_xai.py --skip-warnings                  # 기상특보 API 건너뛰기
```

#### 출력 (JSON, `outputs/explanations_YYYYMMDD.json`)

```json
[
  {
    "item_id": "apple_fuji_box10kg_high",
    "forecast_summary": "1~2문장 요약",
    "forecast_explanation": "2문단 상세 설명",
    "web_sources": [...],
    "_meta": {
      "model": "gpt-4o",
      "web_search_calls": 2,
      "response_id": "resp_..."
    }
  }
]
```

---

## 7. 데이터 갱신 SOP

### 7-1. 매일 갱신 (필수)

`01_DATA/data/full_baseline_extended_*.parquet` 의 **마지막 timestamp = today − 1** 이 되도록 매일 갱신.

#### 갱신해야 할 컬럼

| 도메인 | 컬럼 | 출처 |
|---|---|---|
| 가격 | target, amount | aT 도매시장 통계 |
| 기상 | weather_temp_range, weather_humidity_avg, weather_rain_sum, weather_wind_avg, weather_pressure_avg, weather_sunshine_dur | 기상청 지상관측 (전일 관측값) |
| 거시 (월별 ffill) | bok_base_rate, cpi_growth_rate | 한국은행, 통계청 |
| 에너지 | oil_tax_free_diesel | 한국석유공사 |
| 감성 | news_sentiment_index | 자체 크롤링 + 감성분석 |
| 캘린더 | market_rest | 도매시장 휴장일 캘린더 |

### 7-2. 월별 갱신 (선택)

매월 1일 또는 한국은행/통계청 발표 직후:
- `bok_base_rate` (월 1회 금통위)
- `cpi_growth_rate` (월 1회 통계청)

이 두 값은 운영 코드에서 ffill 되므로 매일 갱신해도 무관.

### 7-3. 신규 품목 추가 절차

1. 도매가 시계열 raw 확보 (최소 2년치 권장)
2. `01_DATA/01_build_baseline.py` (또는 동등 ETL 스크립트) 로 6단계 전처리 실행
3. `meta_baseline.json` 의 `item_station_map` 에 `<신규_item_id>: <주산지 영문명>` 추가
4. **TimesFM ZS**: 추가 학습 없이 즉시 동작 ✅
5. **Chronos2 LoRA**: 정확도 유지 위해 retrain 권장 (8~10분, RTX 5060 Ti)
   ```bash
   # 별도 학습 스크립트 (본 폴더 외부에 보관) 필요
   # 또는 5_12_fin/src/train_chronos2_lora.py 패턴 참조
   ```

### 7-4. 파이프라인 자동화 권장

- **Airflow / cron / Cloud Scheduler** 로 매일 06:00 KST 트리거
- 데이터 갱신 실패 시 알람 (Slack/이메일)
- 예측 결과를 PostgreSQL / Redis / S3 에 저장

---

## 8. 운영 시 주의사항

### 8-1. ★ 데이터 마지막 timestamp 정렬

- `full_baseline` 의 마지막 timestamp = today − 1 이어야 horizon 일자와 일기예보 D+1~D+10 매핑이 정확
- 매일 갱신 실패하면 모든 예측 날짜가 어긋남

### 8-2. ★ 기상청 발표시각

- **06시 발표 사용 권장** — 중기예보가 D+4~D+10 모두 제공
- 18시 발표 시 D+4 누락 → 코드는 단기 보조 + fallback 처리 (정확도 약간 손실)
- 단기예보는 매 3시간 갱신, 발효 +15분 후 호출 가능

### 8-3. TimesFM Zero-Shot 회귀 방지

- TimesFM 의 RevIN 내부 정규화 사용 → **입력 데이터를 외부에서 정규화하지 말 것**
- raw price (원 단위) 그대로 전달

### 8-4. AutoGluon predictor 로드

- `predictor.pkl` 은 Python pickle. **버전 호환 매우 민감**
- 학습 시 환경과 운영 환경의 `autogluon.timeseries` / `torch` 버전 일치 확인
- 버전 충돌 시 retrain (8~10분)

### 8-5. 일기예보 호출 실패 대응

- `02_PROBABILISTIC_FORECAST/predict_example.py` 의 `make_known()` 은 자동 fallback 구조
- API 실패 시 → `lookup` 모드 (full_baseline 의 마지막 관측치 ffill) 로 자동 전환
- 로그에 `[make_known] forecast 모드 실패: ... → lookup 폴백` 출력 → 모니터링 알람 권장

### 8-6. 한글 인코딩 (Windows + bash)

```bash
# 권장 환경변수
export PYTHONNOUSERSITE=1
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1
python -X utf8 <script>.py
```

---

## 9. 트러블슈팅

### 9-1. "ModuleNotFoundError: No module named 'covariate_builder'"

- 원인: 02_PROBABILISTIC_FORECAST 에서 05_WEATHER_FORECAST import path 누락
- 해결: 환경변수 `PYTHONPATH=<final_handoff>/05_WEATHER_FORECAST` 추가 또는 코드의 `sys.path.insert(...)` 가 정상 실행되는지 확인

### 9-2. "predictor.pkl 로드 오류 (KeyError / AttributeError)"

- 원인: AutoGluon 버전 mismatch
- 해결: `pip install autogluon.timeseries==1.5.0` 정확한 버전 설치. torch 도 호환 확인.

### 9-3. "TimesFM HuggingFace 다운로드 실패"

- 원인: 네트워크 / 토큰
- 해결:
  1. HuggingFace 계정 토큰 환경변수: `HF_TOKEN=hf_...`
  2. 또는 사전 다운로드:
     ```python
     from huggingface_hub import snapshot_download
     snapshot_download("google/timesfm-2.5-200m-pytorch")
     ```

### 9-4. "기상청 API non-JSON response"

- 원인: 인증키 만료 / 잘못된 base_time / Rate limit 초과
- 해결: data.go.kr 마이페이지에서 키 상태 확인 / 잠시 대기 후 재호출

### 9-5. "한글 폰트 깨짐 (시각화)"

- 원인: matplotlib 기본 폰트가 한글 미지원
- 해결:
  ```python
  import matplotlib.pyplot as plt
  plt.rcParams["font.family"] = "Malgun Gothic"   # Windows
  # plt.rcParams["font.family"] = "AppleGothic"   # macOS
  ```

### 9-6. "make_known 결과 NaN"

- 원인: full_baseline 의 weather_temp_range 컬럼 모두 NaN (드물지만 가능)
- 해결: `covariate_builder.py` 가 historical 최근 30일 평균 fallback (없으면 10.0) 사용. 단 학습 시 NaN 있으면 retrain 권장.

### 9-7. Chronos2 LoRA "checkpoint not found"

- 원인: `model/models/Chronos2LoRA_baseline/W0/fine-tuned-ckpt/` 가 .gitignore 되어 있을 수 있음
- 해결: 본 폴더 인계 시 model/ 디렉토리 전체 동봉. git LFS 사용 시 `git lfs pull` 확인.

---

## 10. 모니터링 권장

### 10-1. 예측 정확도 모니터링

| 지표 | 측정 방법 | 알람 임계값 |
|---|---|---|
| **Chronos2 WQL** | 매일 D-10 ~ D-1 의 실제값 vs 예측 분위 비교 | WQL > 0.18 (학습 기준 +40%) |
| **Chronos2 PICP@80** | 80% 구간 안에 실제값 들어간 비율 | PICP@80 < 0.7 (목표 0.8) |
| **TimesFM MAE** | 매일 D-3 ~ D-1 의 실제값 vs 예측 비교 | MAE > 4,500 (학습 +50%) |
| **TimesFM MAPE** | MAPE 슬라이딩 윈도우 30일 | MAPE > 18% |

### 10-2. 시스템 모니터링

| 지표 | 알람 |
|---|---|
| 일기예보 API 호출 실패율 | > 5% |
| `make_known` fallback 발생 | > 1회/일 |
| 예측 실행 시간 | > 5분 (정상 ~2분) |
| `full_baseline` 마지막 timestamp 정합 | today − 1 아니면 알람 |
| OpenAI 토큰 사용량 | 일 한도 80% 도달 |

### 10-3. Drift 감지

- 매일 D+1 점예측 vs 다음 날 실측 → MAPE 추세 모니터링
- 한 작물군 (예: spinach) 의 MAPE 가 30일 이동평균 +50% 도달 → ML 팀에 retrain 요청 알림
- 분기별 자동 retrain (Chronos2 LoRA) 권장

---

## 11. 부록: 핵심 데이터 스키마

### 11-1. `01_DATA/data/full_baseline.parquet`

| 컬럼 | 타입 | 분류 | 단위 | 설명 |
|---|---|---|---|---|
| item_id (index) | str | - | - | 품목 ID (46개) |
| timestamp (index) | Timestamp | - | - | 일별 (KST) |
| **target** | float | - | 원 | 일별 도매 평균가 |
| **amount** | float | past | kg or 건 | 거래량 (품목별 단위 상이) |
| **market_rest** | int | **known** | 0/1 | 도매시장 휴장 여부 |
| **weather_temp_range** | float | **known** | °C | 일교차 (TMX-TMN) |
| **weather_sunshine_dur** | float | **known** | hr | 일조시간 |
| **bok_base_rate** | float | **known** | % | 한국은행 기준금리 |
| **cpi_growth_rate** | float | **known** | % | CPI 전년 동월대비 증가율 |
| **weather_humidity_avg** | float | past | % | 일 평균 상대습도 |
| **weather_rain_sum** | float | past | mm | 일 누적 강수량 |
| **weather_wind_avg** | float | past | m/s | 일 평균 풍속 |
| **weather_pressure_avg** | float | past | hPa | 일 평균 해면기압 |
| **oil_tax_free_diesel** | float | past | 원/L | 면세 경유가 |
| **news_sentiment_index** | float | past | -1~1 | 농업 뉴스 감성지수 |

### 11-2. `01_DATA/meta_baseline.json`

```json
{
  "variant": "baseline",
  "known_covariates": ["market_rest", "weather_temp_range", "weather_sunshine_dur",
                       "bok_base_rate", "cpi_growth_rate"],
  "past_covariates": ["amount", "oil_tax_free_diesel", "weather_rain_sum",
                      "weather_wind_avg", "weather_humidity_avg",
                      "weather_pressure_avg", "news_sentiment_index"],
  "item_station_map": {
    "apple_fuji_box10kg_high": "jecheon",
    "cabbage_net8kg_high": "jeju",
    ...
  },
  "train_end": "2024-02-29",
  "notes": "..."
}
```

### 11-3. `01_DATA/data/static_baseline.parquet`

| 컬럼 | 의미 |
|---|---|
| item_id (index) | 품목 ID |
| crop | 작물명 (13개) |
| grade | 등급 (high/mid/low/premium) |
| weather_station | 매핑된 주산지 |

### 11-4. Chronos2 출력 형식

```
                                       0.1     0.2     ...    0.9    mean
item_id                  timestamp
apple_fuji_box10kg_high  2026-05-16   41443   44021   ...   60377   50230
                         2026-05-17   41267   43985   ...   63360   51890
                         ...
```

### 11-5. TimesFM 출력 형식

```
                  item_id   timestamp  y_pred
apple_fuji_box10kg_high  2026-05-16  50492.0
apple_fuji_box10kg_high  2026-05-17  55016.0
apple_fuji_box10kg_high  2026-05-18  58789.0
cabbage_net8kg_high      2026-05-16   8201.0
...
```

---

## 핵심 체크리스트 (인계 직후)

- [ ] Python 3.10/3.11 환경 구축
- [ ] conda 환경 분리 (prob_forecast / point_forecast / xai)
- [ ] 각 모듈 requirements.txt 설치
- [ ] 기상청 API 키 2개 발급 + `.env` 저장
- [ ] (선택) OpenAI 키 발급 + `.env` 저장
- [ ] 단위 검증:
  - [ ] `python 05_WEATHER_FORECAST/stn_grid_map.py` → 산지 18 매핑 OK
  - [ ] `python 05_WEATHER_FORECAST/weather_fetcher.py` → 단기/중기 응답 정상
  - [ ] `python 05_WEATHER_FORECAST/covariate_builder.py` → 460행 NaN 없음
- [ ] 통합 검증:
  - [ ] `python 02_PROBABILISTIC_FORECAST/predict_example.py` → 460행 출력
  - [ ] `python 03_POINT_FORECAST/predict_example.py` → 138행 출력
- [ ] XAI dry-run: `python 04_XAI_EXPLAINER/run_xai.py --limit 1 --dry-run`
- [ ] ETL 파이프라인 구성 (매일 06:00 자동 트리거)
- [ ] DB/캐시 출력 스키마 정의
- [ ] 모니터링 알람 설정
- [ ] 운영 문서화 (런북, 비상 대응)

---

## 문의

- ML / 모델 관련: `00_INTEGRATED_REPORT.md`, `EXPERIMENT_REPORT.md` 참조
- 데이터 스키마: `01_DATA/README.md`, `01_DATA/dataset_description_*.md`
- 일기예보 API: `05_WEATHER_FORECAST/README.md`, `99_REFERENCES/`
- 검증 로그: `VERIFICATION.md`

질문이 본 매뉴얼에서 답을 찾지 못하면 ML 팀에 문의.
