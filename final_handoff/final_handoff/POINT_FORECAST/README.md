# POINT_FORECAST — TimesFM 2.5 Zero-Shot (3일 점예측)

## 한 줄 요약

46개 농산물 품목의 **3일 점예측 (mean)** 모델. Google **TimesFM 2.5** (200M, foundation model) 을 zero-shot 으로 사용. 추가 학습 없음.

| 지표 | 값 |
|---|---|
| RMSE ↓ | 3,364.96 |
| MAE ↓ | 2,994.97 |
| MAPE (%) ↓ | 12.64 |
| **MASE ↓** | **0.806** |

비교한 4 모델 중 1위. 선정 근거·평가 과정은 최상위 [`../INTEGRATED_REPORT.md`](../INTEGRATED_REPORT.md) §4 참조.

## 빠른 시작

```bash
pip install -r requirements.txt
python predict_example.py
```

최초 실행 시 HuggingFace 에서 모델 가중치 (≈200MB) 자동 다운로드. 이후 캐싱됨.

## 추론 호출 (3줄)

```python
import timesfm
from timesfm.configs import ForecastConfig

m = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
m.compile(ForecastConfig(max_context=384, max_horizon=32,
                          normalize_inputs=True, return_backcast=False))
point, _ = m.forecast(horizon=3, inputs=[item_series_numpy])  # list of 1D np.float32
# point.shape = (N_items, 3)
```

## 파일 구조

```
POINT_FORECAST/
├── README.md                  # ← 본 파일
├── dataset_description.md     # 데이터 컬럼 설명
├── predict_example.py         # 한 번 호출하면 끝 (3일 점예측 출력)
├── requirements.txt           # timesfm + torch
└── data/
    ├── full_baseline.parquet  # PROBABILISTIC_FORECAST 와 동일 데이터 (target 만 사용)
    └── meta_baseline.json     # (참고용)
```

> **모델 가중치는 폴더 안에 포함되지 않음.** HuggingFace `google/timesfm-2.5-200m-pytorch` 에서 자동 다운로드. 인터넷이 없는 환경이면 `~/.cache/huggingface/hub/` 의 캐시를 사전 복사해야 함.

## 주의

- 본 모델은 **univariate** — covariate (휴장·기상·금리 등) 사용 안 함. 가격 시리즈만으로 추론.
- 만약 신뢰구간(80%, 60%) 도 필요하면 [`../PROBABILISTIC_FORECAST/`](../PROBABILISTIC_FORECAST/) 의 Chronos2 LoRA 모델 사용. 점예측 단일값만 필요하면 본 모델이 더 정확.
- TimesFM 의 RevIN 내부 정규화가 적용되므로 **입력 데이터를 외부에서 정규화하지 말 것** (raw price 그대로 전달).
