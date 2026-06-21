# POINT_FORECAST 데이터셋 구성 설명

본 폴더의 점예측 모델 (TimesFM 2.5 Zero-Shot) 은 **target 시리즈만** 사용 (univariate).
covariate 컬럼들은 `PROBABILISTIC_FORECAST` 와 데이터 일관성 유지를 위해 함께 제공.

## 파일 목록

| 파일 | 크기 | 설명 |
|---|---|---|
| `data/full_baseline.parquet` | 1.9 MB | 메인 시계열 데이터 (46 품목 × 일별) |
| `data/meta_baseline.json` | 2.9 KB | covariate 분류 명세 (참고용) |

## 핵심 사용 컬럼

| 컬럼 | 타입 | 단위 | 설명 |
|---|---|---|---|
| `item_id` | string | — | `{작물}_{규격}_{등급}` |
| `timestamp` | datetime | day | 일별 (KST) |
| **`target`** | float64 | 원(₩) | 농산물 도매 가격 — **점예측 모델이 사용하는 유일한 컬럼** |

추가로 12개 covariate 컬럼이 함께 들어있으나 TimesFM ZS 추론에는 사용되지 않음. 자세한 컬럼 정의는 [PROBABILISTIC_FORECAST/dataset_description.md](../PROBABILISTIC_FORECAST/dataset_description.md) 참조.

## Train / Test 분할

- Train end: `2024-02-29`
- Test: `2024-03-01 ~ 2026-02-28` (731일, 평가용 — 운영 시 의미 없음)

## 추론 입력 형식

각 품목별 최근 `CONTEXT_LEN=384` 일치 target 값을 1D numpy float32 배열로 모델에 전달.
`predict_example.py` 의 `inputs = list[np.ndarray]` 형태 그대로 사용.
