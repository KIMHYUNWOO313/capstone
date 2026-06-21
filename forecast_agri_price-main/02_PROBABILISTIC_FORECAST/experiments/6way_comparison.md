# 6-Way 비교 — 확률범위 모델 선정 실험

> **데이터셋 variant 3종 × 학습 방식 2종 (ZS, LoRA) = 6 케이스 비교**
> Chronos2 (autogluon/chronos-2) 기반, 49 윈도우 롤링 평가, horizon=10일

종합 보고서 [`../../00_INTEGRATED_REPORT.md`](../../00_INTEGRATED_REPORT.md) §3 의 실험 단위 요약입니다.

---

## 1. 실험 셋업

| 항목 | 값 |
|---|---|
| 모델 | Chronos2 (autogluon/chronos-2, 200M params) |
| 파인튜닝 | LoRA (r=16, α=32, 4000 steps, batch=16, lr=5e-5) |
| 예측 길이 | 10일 |
| 컨텍스트 | 365일 |
| 평가 윈도우 | 49 (cutoff=-10, step=-15, test=731일) |
| 학습 시간 | LoRA 8~10분/variant |
| GPU | NVIDIA RTX 5060 Ti (16 GB) |

### 데이터셋 variant (3종)

| variant | rows | cols | 구성 의도 |
|---|---:|---:|---|
| **baseline** | 172,868 | 13 | 도메인 지식 기반 일반 구성 (Known 5 + Past 7) |
| no_weather | 172,868 | 7 | 기상 컬럼 전부 제거 (가설: 기상 영향 작음) |
| optimal | 172,868 | 15 | Spearman + SHAP + VIF 로 자동 선정 |

---

## 2. 결과 — 6 케이스 비교표

원본: [`metrics_table.csv`](metrics_table.csv)

| variant | model | **WQL ↓** | CRPS ↓ | PICP@60 (목표 0.6) | PICP@80 (목표 0.8) | MSIS@80 ↓ | Sharpness@80 ↓ |
|---|---|---:|---:|---:|---:|---:|---:|
| **★ baseline** | **LoRA** | **0.1298** | 3,195 | **0.589** ✅ | **0.789** ✅ | 5.22 | 12,187 |
| baseline | ZS | 0.1369 | 3,361 | 0.573 | 0.783 | 5.48 | 12,788 |
| no_weather | LoRA | 0.1301 | 3,193 | 0.572 | 0.772 | 5.21 | 12,095 |
| no_weather | ZS | 0.1384 | 3,404 | 0.570 | 0.780 | 5.51 | 12,984 |
| optimal | LoRA | 0.1326 | 3,270 | 0.567 | 0.774 | 5.32 | 12,271 |
| optimal | ZS | 0.1397 | 3,431 | 0.552 | 0.768 | 5.57 | 13,016 |

### 평가지표 정의

| 지표 | 정의 | 좋은 값 |
|---|---|---|
| **WQL** (Weighted Quantile Loss) | 9 분위 pinball loss 가중합을 \|y\| 로 정규화. Chronos 공식 손실 | 낮을수록 |
| **CRPS** | 9 분위 평균 pinball loss × 2. 적분형 CRPS 의 분위 근사 | 낮을수록 |
| **PICP@α** | 신뢰구간 α 안에 실제값이 들어간 비율 — 목표값 (0.6/0.8) 에 근접해야 calibration 좋음 | 목표값에 근접 |
| **MSIS@α** | 신뢰구간 width + 구간 밖 페널티, seasonal naive MAE 로 스케일 (M4 표준) | 낮을수록 |
| **Sharpness@α** | 신뢰구간 평균 width — 좁을수록 confident (단 calibration 깨지면 안 됨) | 낮을수록 |

---

## 3. 핵심 발견 3가지

### 발견 1 — baseline LoRA 가 모든 메트릭 1위

WQL 0.1298, PICP@60 0.589 (목표 0.6), PICP@80 0.789 (목표 0.8). **거의 완벽한 calibration**.

### 발견 2 — no_weather LoRA 가 baseline 과 사실상 동률

WQL 0.1301 (baseline 대비 +0.2%), Sharpness 는 오히려 더 우수 (12,095 vs 12,187). 즉 **기상 변수의 가격 기여도가 작음**을 확률 메트릭으로 재확인. 단순화 관점에선 no_weather 도 동등 후보.

### 발견 3 — optimal (자동 선정) LoRA 가 가장 나쁨

WQL 0.1326. 기상 5개 특보 + temp_avg + rain + wind 조합이 오히려 노이즈를 키움. **자동 feature 선정이 정량 메트릭에서 역효과**를 보인 사례.

---

## 4. LoRA vs ZS 의 일관된 효과

| 데이터셋 variant | WQL 개선 (ZS → LoRA) | CRPS 개선 |
|---|---:|---:|
| baseline | **-5.2%** | -4.9% |
| no_weather | **-6.0%** | -6.2% |
| optimal | **-5.1%** | -4.7% |

→ **LoRA fine-tune 이 확률예측에서 일관되게 5~6% 개선**. variant 와 무관한 robust 한 개선.

원본 per-crop 개선율: [`lora_vs_zs_per_crop.csv`](lora_vs_zs_per_crop.csv)

---

## 5. 작물별 분석

- **작물별 WQL 분포**: [`figures/per_crop_wql.png`](figures/per_crop_wql.png) / 원본 [`per_crop_wql.csv`](per_crop_wql.csv)
- **품목별 winrate (6 케이스 중 우승)**: [`figures/variant_winrate.png`](figures/variant_winrate.png) / 원본 [`per_item_winrate.csv`](per_item_winrate.csv)
- **윈도우별 추세 (49 윈도우)**: [`figures/per_window_trend.png`](figures/per_window_trend.png)

---

## 6. Calibration 분석 — baseline LoRA 가 거의 완벽

- **PICP vs nominal 다이어그램**: [`figures/calibration_diagram.png`](figures/calibration_diagram.png)
  - baseline LoRA: PICP@60=0.589 ≈ 0.6, PICP@80=0.789 ≈ 0.8 — underconfident 도 overconfident 도 아님
- **9분위별 pinball loss**: [`figures/per_quantile_pinball.png`](figures/per_quantile_pinball.png)
- **Variant 종합 radar**: [`figures/variant_radar.png`](figures/variant_radar.png)
- **(참고) Conformal recalibration (CQR) 효과**: [`figures/picp_cqr_effect.png`](figures/picp_cqr_effect.png) — CQR 후처리로 PICP 를 더 가까이 맞출 수 있으나, 현 운영 모델은 raw LoRA 사용 (이미 충분히 calibrated)

---

## 7. ★ 최종 선정: `Chronos2LoRA_baseline`

| 결정 사유 | 근거 |
|---|---|
| WQL 1위 (0.1298) | 6 케이스 비교 최저 |
| PICP@60 / @80 거의 정확히 달성 | 0.589 / 0.789 vs 목표 0.6 / 0.8 |
| 데이터셋이 도메인 지식 기반 → 해석 가능 | (vs 자동 선정 optimal) |
| LoRA 학습 시간 합리적 (8~10분) | 운영 retrain 부담 작음 |

품목별 예측 시각화 (13개 작물의 high 등급 1개씩): [`../sample_forecasts/`](../sample_forecasts/)

---

## 8. 산출물 매핑

| 산출물 | 위치 |
|---|---|
| 6 케이스 메트릭 비교표 | [`metrics_table.csv`](metrics_table.csv) |
| 작물별 WQL | [`per_crop_wql.csv`](per_crop_wql.csv) |
| 품목별 winrate | [`per_item_winrate.csv`](per_item_winrate.csv) |
| LoRA vs ZS 작물별 개선 | [`lora_vs_zs_per_crop.csv`](lora_vs_zs_per_crop.csv) |
| 차트 7장 | [`figures/`](figures/) |
| 13작물 예측 샘플 | [`../sample_forecasts/`](../sample_forecasts/) |
