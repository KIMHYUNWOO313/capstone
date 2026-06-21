# PROBABILISTIC_FORECAST — Chronos2 LoRA (10일 확률범위 예측)

## 한 줄 요약

46개 농산물 품목의 **10일 확률범위 예측** (9분위 + mean) 을 수행하는 AutoGluon 기반 Chronos2 LoRA 모델 (variant: `baseline`).

| 지표 | 값 |
|---|---|
| WQL ↓ | **0.1298** |
| PICP@60 (목표 0.6) | **0.589** ✅ |
| PICP@80 (목표 0.8) | **0.789** ✅ |
| Sharpness@80 ↓ | 12,187 |

선정 근거·평가 과정·다른 후보와 비교는 최상위 [`../INTEGRATED_REPORT.md`](../INTEGRATED_REPORT.md) 참조.

## 빠른 시작

```bash
pip install -r requirements.txt
python predict_example.py
```

GPU 없이도 동작 (CPU 대비 GPU 5-10배 빠름).

## 출력 형식

`predictor.predict()` 결과 컬럼:

| 컬럼 | 의미 |
|---|---|
| `0.1` ~ `0.9` | 9개 분위 예측값 |
| `mean` | 평균 예측 |

- **80% 신뢰구간**: `0.1` ~ `0.9`
- **60% 신뢰구간**: `0.2` ~ `0.8`
- **중앙값 (점예측 대용)**: `0.5`

## 파일 구조

```
PROBABILISTIC_FORECAST/
├── README.md                  # ← 본 파일
├── dataset_description.md     # 데이터 컬럼·item_id·주산지 매핑 등 상세 명세
├── predict_example.py         # 한 번 호출하면 끝 (10일 확률범위 출력)
├── requirements.txt           # autogluon.timeseries==1.5.0 + torch
├── data/
│   ├── full_baseline.parquet   # 메인 시계열 (172,868 rows × 13 cols)
│   ├── static_baseline.parquet # 품목별 정적 속성
│   └── meta_baseline.json      # known/past covariate 분류
└── model/                       # AutoGluon TimeSeriesPredictor 통째
    ├── predictor.pkl
    ├── learner.pkl
    └── models/
        └── Chronos2LoRA_baseline/   # ← 본 모델 (LoRA r=16, α=32)
            └── W0/fine-tuned-ckpt/
```

## 운영 시 known covariate 갱신

`predict_example.py` 의 `make_known()` 함수는 평가용으로 lookup 테이블 사용. 실제 운영에서는 다음 외부 데이터 소스 연결 필요:

| known covariate | 외부 데이터 소스 |
|---|---|
| `market_rest` | 도매시장 휴장일 캘린더 |
| `weather_temp_range` | 기상청 7~10일 예보 (품목별 주산지) |
| `weather_sunshine_dur` | 기상청 예보 |
| `bok_base_rate` | 한국은행 금통위 발표 (월 1회, ffill) |
| `cpi_growth_rate` | 통계청 월별 발표 (ffill) |

품목별 주산지 매핑은 `dataset_description.md` §6 참조.
