# 4-Way 비교 — 점예측 모델 기본 비교 실험

> **TimesFM 2.5 vs IBM Granite TTM × {ZS, Fine-tune} = 4 케이스**
> 49 윈도우 롤링 평가, horizon=3일

종합 보고서 [`../../00_INTEGRATED_REPORT.md`](../../00_INTEGRATED_REPORT.md) §4 의 실험 단위 요약입니다.

> 이 4-way 외에 다변량 활용 ablation 까지 합친 **7-way 비교** 는 [`7way_ablation.md`](7way_ablation.md) 에서 다룹니다.

---

## 1. 실험 셋업

| 항목 | 값 |
|---|---|
| 비교 모델 | TimesFM 2.5 (200M, Google) + IBM Granite TTM (R2, ~5M) |
| 평가 방식 | ZS (zero-shot) + Fine-tune 각각 |
| 예측 길이 | 3일 |
| 컨텍스트 | 384일 (TimesFM, 32배수) / 512일 (TTM) |
| 평가 윈도우 | 49 (cutoff=-3, step=-15, test=731일) |
| Fine-tune | TimesFM: PEFT-LoRA (r=8, α=16, 3 epoch, lr=1e-4) / TTM: backbone freeze + decoder/head 학습 |
| GPU | NVIDIA RTX 5060 Ti (16 GB) |

---

## 2. 결과 — 4 케이스 비교표

원본: [`metrics_table_4way.csv`](metrics_table_4way.csv)

| model | RMSE | MAE | MAPE (%) | **MASE ↓** |
|---|---:|---:|---:|---:|
| **★ TimesFM 2.5 ZS** | **3,364.96** | **2,994.97** | **12.64** | **0.806** |
| TimesFM 2.5 LoRA (best: r=8, α=16, lr=1e-4) | 3,426.55 | 3,056.77 | 13.09 | 0.811 |
| TTM FT (decoder/head fine-tune) | 3,800.45 | 3,405.19 | 15.23 | 0.931 |
| TTM ZS | 8,198.51 | 7,884.90 | 44.16 | 2.184 |

### 평가지표 정의

| 지표 | 정의 | 좋은 값 |
|---|---|---|
| **RMSE** | √(평균(예측−실제)²). 단위: 원 | 낮을수록 |
| **MAE** | 평균(|예측−실제|). 단위: 원 | 낮을수록 |
| **MAPE** | 평균(|예측−실제| / |실제|) × 100. 단위: % | 낮을수록 |
| **MASE** | MAE 를 train 기간 7일-seasonal naive MAE 로 나눔. 1.0 미만 = naive 보다 우수 | 낮을수록 |

---

## 3. Fine-tuning 개선율

원본: [`improvement_pct.csv`](improvement_pct.csv)

| model | RMSE Δ | MAE Δ | MAPE Δ | MASE Δ |
|---|---:|---:|---:|---:|
| TimesFM (ZS → LoRA) | **-1.83%** | -2.06% | -3.57% | **-0.56%** |
| TTM (ZS → FT) | +53.64% | +56.81% | +65.51% | **+57.38%** |

→ **TimesFM 의 LoRA 는 ZS 를 개선 못 함**. TTM 은 큰 폭 개선이지만 ZS 자체가 매우 약했기 때문 (자세한 인과 분석은 §5).

---

## 4. 핵심 발견 4가지

### 발견 1 — TimesFM 2.5 ZS 가 본 데이터에서 매우 강력

농산물 도메인 명시적 학습 없이 MASE 0.806 (1.0 미만 = seasonal naive 보다 우수). foundation model 의 일반화 성능 입증.

### 발견 2 — LoRA 가 ZS 를 개선 못 함 (-0.56%)

본 데이터 규모 (46 시리즈 × 2000 random windows × 3 epoch) 로는 ZS 의 보편적 패턴을 능가 못 함. 더 큰 데이터·더 많은 epoch 시도 가치 있으나 ROI 불확실.

### 발견 3 — TTM 의 큰 개선 폭 (+57%) 은 base 약함의 정상화

TTM-R2 base 는 농산물에 사전학습 없음 + `decoder_mode="mix_channel"` 활성화 시 channel mixer 가 random init. fine-tune 으로 정상화 (MASE 2.18 → 0.93). 절대 성능 보단 차이의 폭이 큰 것뿐.

> 이 사실은 곧 **"데이터에 학습 가능한 신호가 풍부함"** 의 증거이기도 함. [`7way_ablation.md`](7way_ablation.md) §5 참조.

### 발견 4 — 품목별 우승은 ZS 24 / LoRA 22

46 품목을 거의 반반 차지. TTM 은 0 품목. 품목별 ZS/LoRA 분기 라우팅 시 추가 이득 가능성 있음. 원본 [`per_crop_mase_4way.csv`](per_crop_mase_4way.csv).

---

## 5. 시각자료

| 시각자료 | 위치 |
|---|---|
| 4-way 막대 차트 | [`figures/model_comparison_bar.png`](figures/model_comparison_bar.png) |
| 품목별 우승 분포 | [`figures/per_crop_winner.png`](figures/per_crop_winner.png) |
| 3-모델 점예측 비교 (8 품목 grid) | [`../sample_forecasts/point_compare_grid_8items.png`](../sample_forecasts/point_compare_grid_8items.png) |
| 개별 품목 비교 8장 | [`../sample_forecasts/point_*_high.png`](../sample_forecasts/) |

**특징적 관찰** (sample_forecasts):
- **양파(onion)**: TimesFM ZS/LoRA 가 actual 에 거의 일치, TTM FT 만 잘못된 방향 → ZS 우세 케이스의 전형
- **사과(apple)**: 가격 변동성 큰 구간에서 TTM FT 가 단기 상승 추세를 더 잘 잡아냄 → 모든 케이스에서 ZS 우승은 아님
- **시금치(spinach)**: TimesFM ZS/LoRA 거의 동일, TTM 은 일관되게 위로 편향

---

## 6. ★ 최종 선정: TimesFM 2.5 Zero-Shot

| 결정 사유 | 근거 |
|---|---|
| MASE 최저 (0.806) | 4-way 비교 1위 |
| 학습 비용 0 | adapter / fine-tune 파일 관리 불필요 |
| 운영 단순 | base 모델만 다운로드, retrain 불필요 |
| 신규 품목 / 도메인 즉시 대응 | foundation model 의 일반화 활용 |
| 결과 재현성 | 학습 단계 없음 → 환경별 결과 편차 없음 |

> **방어 근거**: "covariate 12개 갖고도 단변량 ZS 만 썼다" 는 비판에 대해 5가지 다변량 활용 ablation 으로 정량 검증. [`7way_ablation.md`](7way_ablation.md) 참조.

---

## 7. 산출물 매핑

| 산출물 | 위치 |
|---|---|
| 4 케이스 메트릭 비교 | [`metrics_table_4way.csv`](metrics_table_4way.csv) |
| 품목별 MASE (4-way) | [`per_crop_mase_4way.csv`](per_crop_mase_4way.csv) |
| Fine-tune 개선율 | [`improvement_pct.csv`](improvement_pct.csv) |
| 그리드 서치 상세 | [`grid_search.md`](grid_search.md) |
| 차트 | [`figures/model_comparison_bar.png`](figures/model_comparison_bar.png), [`figures/per_crop_winner.png`](figures/per_crop_winner.png) |
| 예측 샘플 | [`../sample_forecasts/`](../sample_forecasts/) |
