# 02_PROBABILISTIC_FORECAST — Chronos2 LoRA (10일 확률범위 예측)

> **모델 선정 결과** — 6 케이스 비교 (3 데이터셋 variant × {ZS, LoRA}) 에서 1위
> 자세한 실험·근거: [`experiments/6way_comparison.md`](experiments/6way_comparison.md)

---

## 1. 한 줄 요약

46개 농산물 품목의 **10일 확률범위 예측** (9분위 + mean) 을 수행하는 AutoGluon 기반 Chronos2 LoRA 모델 (variant: `baseline`).

| 지표 | 값 | 비고 |
|---|---|---|
| **WQL ↓** | **0.1298** | 6 케이스 비교 최저 |
| **PICP@60** (목표 0.6) | **0.589** ✅ | 거의 완벽한 calibration |
| **PICP@80** (목표 0.8) | **0.789** ✅ | 거의 완벽한 calibration |
| Sharpness@80 ↓ | 12,187 | |

---

## 2. 모델·학습 설정

| 항목 | 값 |
|---|---|
| Base 모델 | Chronos2 (autogluon/chronos-2, 200M) |
| 파인튜닝 | LoRA (r=16, α=32, 4000 steps, batch=16, lr=5e-5) |
| 예측 길이 | 10일 |
| 컨텍스트 | 365일 |
| 평가 윈도우 | 49 (cutoff=-10, step=-15, test=731일) |
| 학습 시간 | 8~10분 (RTX 5060 Ti) |
| 사용 변수 | target + Known 5 + Past 7 (도메인 지식 기반 baseline variant) |

---

## 3. 폴더 구조

```
02_PROBABILISTIC_FORECAST/
├── README.md                       ← 본 파일 (모델 개요·선정 결과)
├── predict_example.py              ← 즉시 실행 가능한 추론 예제
├── requirements.txt                ← autogluon.timeseries==1.5.0
│
├── experiments/                    ★ 실험·비교·근거
│   ├── 6way_comparison.md          ← variant 3 × {ZS, LoRA} 6 케이스 비교 narrative
│   ├── metrics_table.csv           ← 6 케이스 메트릭 표 (WQL/CRPS/PICP/MSIS/Sharpness)
│   ├── per_crop_wql.csv            ← 작물별 WQL
│   ├── per_item_winrate.csv        ← 품목별 winrate
│   ├── lora_vs_zs_per_crop.csv     ← LoRA vs ZS 작물별 개선율
│   └── figures/                    ← 7장 차트
│       ├── calibration_diagram.png
│       ├── variant_radar.png
│       ├── variant_winrate.png
│       ├── per_crop_wql.png
│       ├── per_quantile_pinball.png
│       ├── per_window_trend.png
│       └── picp_cqr_effect.png
│
├── sample_forecasts/               ← 13작물 high 등급 예측 시각화 (PNG)
│
└── model/                          ← AutoGluon TimeSeriesPredictor (즉시 로드)
    ├── predictor.pkl
    ├── learner.pkl
    └── models/
        └── Chronos2LoRA_baseline/  ← 본 모델 (LoRA r=16, α=32)
            └── W0/fine-tuned-ckpt/
```

> 데이터 파일은 [`../01_DATA/`](../01_DATA/) 에 통합되어 있습니다 (두 모델 공통). `predict_example.py` 는 자동으로 `../01_DATA/` 를 참조합니다.

---

## 4. 빠른 재현

```bash
pip install -r requirements.txt
python predict_example.py
```

GPU 없이도 동작 (CPU 대비 GPU 5~10배 빠름).

### 출력 형식

`predictor.predict()` 결과 컬럼:

| 컬럼 | 의미 |
|---|---|
| `0.1` ~ `0.9` | 9개 분위 예측값 |
| `mean` | 평균 예측 |

- **80% 신뢰구간**: `0.1` ~ `0.9`
- **60% 신뢰구간**: `0.2` ~ `0.8`
- **중앙값 (점예측 대용)**: `0.5`

---

## 5. 운영 시 known covariate 갱신 — **기상청 일기예보 통합 완료**

`predict_example.py` 의 `make_known()` 은 두 모드를 지원:

| 모드 | 동작 | 용도 |
|---|---|---|
| **`forecast`** (기본) | [`05_WEATHER_FORECAST`](../05_WEATHER_FORECAST/) 모듈이 기상청 단기/중기 일기예보 API 로 `weather_temp_range` D+1~D+10 채움. 그 외 4종은 마지막 관측치 ffill | **운영** |
| `lookup` | `full_baseline` 의 실제값을 lookup. 테스트셋 평가나 오프라인 검증용 | 평가 |

기본은 `forecast` 모드이며, API 실패 시 자동으로 `lookup` 으로 폴백합니다.

```bash
python predict_example.py             # 기본: forecast 모드 (기상청 API)
python predict_example.py lookup       # 명시적 lookup (평가용)
```

| known covariate | 운영 시 처리 |
|---|---|
| `market_rest` | 마지막 관측치 ffill (한국천문연구원 특일정보 API 통합 권장) |
| `weather_temp_range` | **기상청 단기예보(D+1~D+3) + 중기예보(D+4~D+10)** ★ |
| `weather_sunshine_dur` | 마지막 관측치 ffill (단기/중기 API 미제공) |
| `bok_base_rate` | 마지막 관측치 ffill (월 단위 변동) |
| `cpi_growth_rate` | 마지막 관측치 ffill (월 단위 변동) |

품목별 주산지 매핑은 [`../01_DATA/meta_baseline.json`](../01_DATA/meta_baseline.json) 의 `item_station_map` 참조 (18개 산지). 산지명 → 격자/구역코드 변환은 [`../05_WEATHER_FORECAST/stn_grid_map.py`](../05_WEATHER_FORECAST/stn_grid_map.py).

> **운영 시 주의**: `full_baseline` 의 마지막 timestamp = today − 1 이어야 horizon 일자와 일기예보 D+1~D+10 매핑이 정확합니다. 매일 데이터 갱신 파이프라인 필수.

---

## 6. 발표 흐름에서의 위치

이 폴더는 **"확률범위 예측 모델은 어떻게 선정되었는가"** 에 답합니다:

1. **데이터 variant 3종 비교** → baseline 채택 (도메인 지식 vs 자동 선정의 함정)
2. **ZS vs LoRA 일관 개선** → LoRA fine-tune 의 의미
3. **거의 완벽한 calibration** → 운영 신뢰구간으로 즉시 활용 가능

→ 발표 슬라이드 작성 시 [`experiments/6way_comparison.md`](experiments/6way_comparison.md) 의 §2 (비교표), §3 (핵심 발견), §6 (calibration 분석) 을 그대로 활용 가능.
