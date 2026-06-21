# 03_POINT_FORECAST — TimesFM 2.5 Zero-Shot (3일 점예측)

> **모델 선정 결과** — 4-way 기본 비교 + 5가지 다변량 활용 ablation 의 **7-way 비교 1위**
> 자세한 실험:
> - [`experiments/4way_comparison.md`](experiments/4way_comparison.md) — 4-way 기본 비교 (TimesFM/TTM × ZS/FT)
> - [`experiments/7way_ablation.md`](experiments/7way_ablation.md) ★ 다변량 활용 시도 + ZS 정당화
> - [`experiments/grid_search.md`](experiments/grid_search.md) — 하이퍼파라미터 탐색

---

## 1. 한 줄 요약

46개 농산물 품목의 **3일 점예측 (mean)** 모델. Google **TimesFM 2.5** (200M, foundation model) 을 zero-shot 으로 사용. 추가 학습 없음.

| 지표 | 값 |
|---|---|
| RMSE ↓ | 3,364.96 |
| MAE ↓ | 2,994.97 |
| MAPE (%) ↓ | 12.64 |
| **MASE ↓** | **0.806** |

**비교한 7가지 모델/방식 중 1위** (단변량 ZS + 단변량 LoRA + 다변량 5가지).

---

## 2. ★ "왜 단변량 ZS 인가" — 다변량 시도 ablation 결과

본 운영 모델이 단변량 ZS 인 것은 **5가지 다변량 활용 방식을 모두 시도** 한 결과:

| 다변량 시도 | MASE | ZS 대비 |
|---|---:|---:|
| **★ TimesFM ZS (단변량, 운영 모델)** | **0.806** | — |
| TimesFM LoRA (단변량 fine-tune) | 0.811 | +0.6% |
| TimesFM xreg + timesfm (다변량 dyn 5) | 0.862 | +6.9% |
| TimesFM xreg + static (다변량 + crop, grade) | 0.865 | +7.3% |
| IBM TTM FT (다변량 dyn 12 + channel mixing) | 0.931 | +15.4% |
| TimesFM timesfm + xreg (잔차 보정) | 1.029 | +27.6% |
| IBM TTM ZS | 2.184 | +170% |

품목별 우승도 단변량 (ZS 22 + LoRA 21) 이 **43/46 = 93%** 차지.

> **데이터셋이 부실해서가 아닙니다.** IBM TTM 이 ZS MASE 2.18 → FT 0.93 으로 **57.4% 비약적 개선** 한 사실이 데이터에 학습 가능한 covariate-target 신호가 충분함을 증명. 다만 TimesFM 2.5 의 사전학습이 그 신호를 이미 잘 흡수해서 추가 학습 (LoRA/FT/xreg) 으로 더 짜낼 여지가 작은 것뿐. 자세한 정량 증거 3가지는 [`experiments/7way_ablation.md`](experiments/7way_ablation.md) §5 참조.

---

## 3. 모델·추론 설정

| 항목 | 값 |
|---|---|
| Base 모델 | TimesFM 2.5 (google/timesfm-2.5-200m-pytorch, 200M) |
| 사용 방식 | Zero-Shot (추가 학습 없음) |
| 컨텍스트 | 384일 (32 배수, ≈1년치) |
| 예측 길이 | 3일 |
| 정규화 | RevIN (모델 내부 자동) |
| 사용 변수 | target 만 (univariate) |
| GPU | 권장 (CUDA 12.x). CPU 도 가능 (5~10배 느림) |

---

## 4. 폴더 구조

```
03_POINT_FORECAST/
├── README.md                       ← 본 파일 (모델 개요·선정 결과)
├── predict_example.py              ← 즉시 실행 가능한 추론 예제
├── requirements.txt                ← timesfm + torch
│
└── experiments/                    ★ 실험·비교·근거
    ├── 4way_comparison.md          ← TimesFM/TTM × {ZS, FT} 4 케이스 비교
    ├── 7way_ablation.md            ★ 다변량 활용 5가지 + ZS 정당화
    ├── grid_search.md              ← LoRA · TTM 하이퍼파라미터 탐색
    ├── metrics_table_4way.csv
    ├── metrics_table_7way.csv
    ├── per_crop_mase_4way.csv
    ├── per_crop_mase_7way.csv
    ├── improvement_pct.csv
    ├── grid_timesfm.csv            ← LoRA 그리드 로그
    ├── grid_ttm.csv                ← TTM 그리드 로그
    ├── per_window_xreg_a.csv       ← xreg 케이스별 윈도우 결과
    ├── per_window_xreg_b.csv
    ├── per_window_xreg_static.csv
    ├── summary_xreg_a.csv          ← xreg 케이스별 요약
    ├── summary_xreg_b.csv
    ├── summary_xreg_static.csv
    └── figures/
        ├── model_comparison_bar.png        ← 4-way 막대
        ├── model_comparison_7way_bar.png   ← 7-way 막대
        ├── per_crop_winner.png             ← 품목별 우승 (4-way)
        └── per_crop_winner_7way.png        ← 품목별 우승 (7-way)

└── sample_forecasts/               ← 3-모델 점예측 비교 (8 품목 + grid)
```

> 모델 가중치는 폴더 안에 포함되지 않음. HuggingFace `google/timesfm-2.5-200m-pytorch` 에서 자동 다운로드. 인터넷 없는 환경이면 `~/.cache/huggingface/hub/` 캐시 사전 복사 필요.
>
> 데이터는 [`../01_DATA/`](../01_DATA/) 통합 폴더 참조 (`predict_example.py` 자동 연결).

---

## 5. 빠른 재현

```bash
pip install -r requirements.txt
python predict_example.py
```

최초 실행 시 HuggingFace 에서 모델 가중치 (≈200MB) 자동 다운로드. 이후 캐싱됨.

### 추론 호출 (3줄)

```python
import timesfm
from timesfm.configs import ForecastConfig

m = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
m.compile(ForecastConfig(max_context=384, max_horizon=32,
                          normalize_inputs=True, return_backcast=False))
point, _ = m.forecast(horizon=3, inputs=[item_series_numpy])  # list of 1D np.float32
# point.shape = (N_items, 3)
```

---

## 6. 주의

- 본 모델은 **univariate** — covariate (휴장·기상·금리 등) 사용 안 함. 가격 시리즈만으로 추론.
- 만약 신뢰구간(80%, 60%) 도 필요하면 [`../02_PROBABILISTIC_FORECAST/`](../02_PROBABILISTIC_FORECAST/) 의 Chronos2 LoRA 모델 사용. 점예측 단일값만 필요하면 본 모델이 더 정확.
- TimesFM 의 RevIN 내부 정규화가 적용되므로 **입력 데이터를 외부에서 정규화하지 말 것** (raw price 그대로 전달).
- **일기예보를 점예측에 활용하지 않습니다** — 49-window 백테스트 결과 단기/중기 일기예보를 dynamic_numerical_covariate(xreg) 로 주입한 두 모드 모두 ZS 대비 MASE +4.8% 악화(oracle/noisy 모두). 자세한 근거는 [`experiments/7way_ablation.md`](experiments/7way_ablation.md) 및 [`../EXPERIMENT_REPORT.md`](../EXPERIMENT_REPORT.md) §3.4 (xreg 백테스트) 참조. 일기예보는 **구간예측(Chronos2)에만** 적용됩니다 ([`../05_WEATHER_FORECAST/`](../05_WEATHER_FORECAST/)).

---

## 7. 발표 흐름에서의 위치

이 폴더는 **"점예측 모델은 어떻게 선정되었는가"** 에 답합니다:

1. **TimesFM vs TTM × ZS/FT 의 4-way 비교** → TimesFM ZS 최고
2. **다변량 ablation 5가지** → 다 시도했으나 ZS 가 우세 = 정당한 선정 근거
3. **데이터 quality 정량 검증** → TTM FT 57.4% 개선·LoRA variant 일관 개선이 신호 풍부함의 증거

→ 발표 슬라이드 작성 시 [`experiments/7way_ablation.md`](experiments/7way_ablation.md) 의 §1 (다변량 시도 5가지), §2 (7-way 비교표), §4 (인과 분석), §5 (데이터 quality 정량 반증) 을 그대로 활용 가능. **§5 와 §6 (방어 근거 메시지) 은 발표 핵심 페이지로 단독 사용 권장**.
