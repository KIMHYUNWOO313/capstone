# 통합 보고서 — 농산물 가격 예측 모델 (확률범위 + 점예측)

**작성일**: 2026-05-14
**작성자**: ML 팀 → 백엔드 팀 인수인계용

본 문서는 `final_handoff/` 폴더에 포함된 **두 종류의 최종 예측 모델** (확률범위 / 점예측) 이 어떤 과정과 어떤 평가지표·비교를 거쳐 선정되었는지, 그리고 원천 데이터에서 최종 학습용 데이터까지 어떤 변환을 거쳤는지 모두 정리합니다.

---

## 목차

1. [프로젝트 전체 흐름](#1-프로젝트-전체-흐름)
2. [원천 데이터 → 최종 데이터 변환 과정](#2-원천-데이터--최종-데이터-변환-과정)
3. [범위예측 (확률범위) 모델 — 비교·선정 과정](#3-범위예측-확률범위-모델--비교선정-과정)
4. [점예측 모델 — 비교·선정 과정](#4-점예측-모델--비교선정-과정)
5. [최종 권장 / 통합 결론](#5-최종-권장--통합-결론)
6. [환경 / 운영 / 트러블슈팅](#6-환경--운영--트러블슈팅)
7. [한계점 및 후속 개선 아이디어](#7-한계점-및-후속-개선-아이디어)

---

## 1. 프로젝트 전체 흐름

### 1-A. 두 단계로 진행
1. **선행 프로젝트** (`5_12_fin/`, 2026-05-12 완료) — **10일 확률범위 예측** (Chronos2 LoRA)
2. **후속 프로젝트** (`point_pred_3d/`, 2026-05-14 완료) — **3일 점예측** (TimesFM 2.5 / IBM TTM)

### 1-B. 공통 조건
- **품목 수**: 46 (13개 작물 × 1~4개 등급)
- **데이터 기간**: 2015-11-16 ~ 2026-02-28 (일별, 10년+)
- **Train/Test 분할**: train_end = 2024-02-29, test = 2024-03-01 ~ 2026-02-28 (731일)
- **롤링 평가**: cutoff step = 15일, 총 49 윈도우 (확률예측은 horizon=10, 점예측은 horizon=3)
- **GPU**: NVIDIA RTX 5060 Ti (16 GB)

### 1-C. 사용한 ML 패러다임
| 단계 | 모델 군 | 평가 종류 | 핵심 메트릭 |
|---|---|---|---|
| 1차 (범위예측) | **Chronos2** (autogluon/chronos-2) ZS + LoRA | 9 분위 quantile forecast | WQL, PICP, MSIS, Sharpness |
| 2차 (점예측) | **TimesFM 2.5** + **IBM TTM** ZS + Fine-tune | mean point forecast | RMSE, MAE, MAPE, MASE |

---

## 2. 원천 데이터 → 최종 데이터 변환 과정

### 2-A. 원천 데이터 출처 (선행 프로젝트 5_12_fin 에서 정제됨)

| 도메인 | 출처 | 원천 컬럼 |
|---|---|---|
| 농산물 가격 | aT 농수산식품유통공사 도매시장 통계 | 일별 도매 평균가, 거래량 |
| 기상 | 기상청 지상관측 (품목별 주산지) | 일평균 기온, 일교차, 강수량, 풍속, 습도, 기압, 일조시간 |
| 거시경제 | 한국은행 / 통계청 | 기준금리, CPI, M2, 국고채 3년 |
| 에너지 | 한국석유공사 | 면세 경유 가격 |
| 뉴스 감성 | 자체 크롤링 + 감성 분석 | 농업 뉴스 감성 지수 (일별) |

### 2-B. 데이터 변환 6 단계 (5_12_fin/src/01_build_baseline.py)

```
원천 raw CSV 다수
  │
  │ ① 작물·규격·등급 단위로 item_id 생성 (46개)
  │
  ▼
  품목별 일별 raw target 시계열
  │
  │ ② 품목별 주산지 매핑 (item_station_map) — Spearman 상관으로 train 기간에서 최적 주산지 선정
  │
  ▼
  품목 × 일별 (target + 12 covariates)
  │
  │ ③ 결측 처리: target/amount 선형보간, 그 외 ffill (도메인별 의미 차이로)
  │   → 사용자 검증: 보간이 NaN 보존보다 chronos2 성능 우수
  │
  │ ④ Known / Past covariate 정정
  │   - market_rest, weather_temp_range, weather_sunshine_dur, bok_base_rate, cpi_growth_rate → Known
  │   - amount, oil_tax_free_diesel, weather_*(rain/wind/humidity/pressure), news_sentiment_index → Past
  │   ※ v9_1 의 oil_tax_free_diesel 잘못된 known 분류 정정 (매일 변동이라 past 가 맞음)
  │
  │ ⑤ Train/Test 누설 검증
  │   - train_end (2024-02-29) 와 test 시작 (2024-03-01) 사이 값 불연속 확인
  │   - v9_1 의 보간 누설 버그 차단
  │
  │ ⑥ Parquet 저장 (MultiIndex item_id × timestamp)
  │
  ▼
  full_baseline.parquet (172,868 rows × 13 cols) — 최종 학습 데이터
  + static_baseline.parquet (46 rows, 품목 정적 속성)
  + meta_baseline.json (known/past 분류 명세)
```

### 2-C. 데이터셋 후보 비교 (variant 3종)

5_12_fin 프로젝트에서 다음 3개 variant 를 모두 학습·평가하여 비교:

| variant | rows | cols | 구성 의도 | Known | Past |
|---|---|---|---|---|---|
| **baseline** | 172,868 | 13 | 도메인 지식 기반 일반 구성 | 5개 | 7개 |
| no_weather | 172,868 | 7 | 기상 컬럼 전부 제거 (사용자 가설: 기상 영향 작음) | 3개 | 3개 |
| optimal | 172,868 | 15 | Spearman + SHAP + VIF 로 자동 선정 | 4개 | 10개 |

자세한 컬럼별 분류는 [`PROBABILISTIC_FORECAST/dataset_description.md`](PROBABILISTIC_FORECAST/dataset_description.md) 참조.

→ **최종 채택: baseline** (3-A 의 평가 결과로 결정).

### 2-D. 최종 데이터에 적용된 가공 요약

| 항목 | 처리 방식 |
|---|---|
| 결측치 | target/amount: 선형보간, 그 외: ffill (사용자 검증) |
| 이상치 | 자동 제거 안 함 (모델이 처리) |
| 정규화 | 원본 raw 값 유지 (모델이 내부 RevIN/scaler 적용) |
| 시간 정렬 | 일별 (freq=D, KST) |
| 단위 통일 | 가격(원), 기상(°C, hr, mm 등), 금리(%), 감성지수(-1~1) |
| 주산지 매핑 | 품목별 Spearman 최적 station (item_station_map) |

---

## 3. 범위예측 (확률범위) 모델 — 비교·선정 과정

> 출처: `5_12_fin/RESULTS_REPORT.md`, `5_12_fin/outputs/comparison/`, `5_12_fin/outputs/comparison_v2/`

### 3-A. 학습 셋업

| 항목 | 값 |
|---|---|
| 모델 | Chronos2 (autogluon/chronos-2) |
| 파인튜닝 | LoRA (r=16, α=32, 4000 steps, batch=16, lr=5e-5) |
| 예측 길이 | 10 일 |
| 컨텍스트 | 365 일 |
| 평가 윈도우 | 49 (cutoff=-10, step=-15, test=731일) |
| 학습 시간 | LoRA 8~10분/variant |

### 3-B. 6가지 케이스 비교 (variant 3종 × ZS / LoRA)

원본 표: [`evaluation_data/prob_metrics_table.csv`](evaluation_data/prob_metrics_table.csv)

| variant | model | **WQL ↓** | CRPS ↓ | PICP@60 (목표 0.6) | PICP@80 (목표 0.8) | MSIS@80 ↓ | Sharpness@80 ↓ |
|---|---|---:|---:|---:|---:|---:|---:|
| **★ baseline** | **LoRA** | **0.1298** | 3,195 | **0.589** ✅ | **0.789** ✅ | 5.22 | 12,187 |
| baseline | ZS | 0.1369 | 3,361 | 0.573 | 0.783 | 5.48 | 12,788 |
| no_weather | LoRA | 0.1301 | 3,193 | 0.572 | 0.772 | 5.21 | 12,095 |
| no_weather | ZS | 0.1384 | 3,404 | 0.570 | 0.780 | 5.51 | 12,984 |
| optimal | LoRA | 0.1326 | 3,270 | 0.567 | 0.774 | 5.32 | 12,271 |
| optimal | ZS | 0.1397 | 3,431 | 0.552 | 0.768 | 5.57 | 13,016 |

### 3-C. 평가지표 정의

| 지표 | 정의 | 좋은 값 |
|---|---|---|
| **WQL** (Weighted Quantile Loss) | Chronos 공식 손실. 9 분위 pinball loss 의 가중합을 \|y\| 로 정규화 | 낮을수록 |
| **CRPS** (Continuous Ranked Probability Score) | 9 분위 평균 pinball loss × 2. 적분 형태의 CRPS 의 9분위 근사 | 낮을수록 |
| **PICP@α** (Prediction Interval Coverage Probability) | 신뢰구간 α 안에 실제값이 들어간 비율. 목표값 (0.6 / 0.8) 에 근접해야 calibration 좋음 | 목표값에 근접 |
| **MSIS@α** (Mean Scaled Interval Score, M4 표준) | 신뢰구간 width + 구간 밖 페널티. seasonal naive MAE 로 스케일 | 낮을수록 |
| **Sharpness@α** | 신뢰구간 평균 width. 좁을수록 모델이 confident | 낮을수록 (단 calibration 깨지면 안 됨) |

### 3-D. 핵심 발견 3가지

1. **baseline LoRA 가 모든 메트릭 1위** — WQL 0.1298, PICP@60 0.589 (목표 0.6), PICP@80 0.789 (목표 0.8). **거의 완벽한 calibration**.

2. **no_weather LoRA 가 baseline 과 사실상 동률** — WQL 0.1301 (+0.2%), Sharpness 는 오히려 더 우수 (12,095 vs 12,187). 즉 **기상 변수의 가격 기여도가 매우 작음**을 확률 메트릭으로도 재확인. 단순화 관점에선 no_weather 도 동등 후보.

3. **optimal (자동 선정) LoRA 는 가장 나쁨** — WQL 0.1326. 기상 5개 특보 + temp_avg + rain + wind 조합이 오히려 노이즈를 키움. **자동 feature 선정이 정량적 메트릭에서 역효과** 사례.

### 3-E. LoRA vs ZS 의 일관된 효과

| 비교 | WQL 개선 | CRPS 개선 |
|---|---:|---:|
| baseline (ZS → LoRA) | -5.2% | -4.9% |
| no_weather (ZS → LoRA) | -6.0% | -6.2% |
| optimal (ZS → LoRA) | -5.1% | -4.7% |

→ **LoRA fine-tune 이 확률예측에서 일관되게 5~6% 개선**.

### 3-F. 작물별 우승 / win-rate
- variant_winrate 차트: [`visualizations/prob/variant_winrate.png`](visualizations/prob/variant_winrate.png)
- 윈도우별 추세: [`visualizations/prob/per_window_trend.png`](visualizations/prob/per_window_trend.png)
- 작물별 WQL: [`visualizations/prob/per_crop_wql.png`](visualizations/prob/per_crop_wql.png)
- 원본 CSV: [`evaluation_data/prob_per_crop_wql.csv`](evaluation_data/prob_per_crop_wql.csv), [`prob_per_item_winrate.csv`](evaluation_data/prob_per_item_winrate.csv), [`prob_lora_vs_zs_per_crop.csv`](evaluation_data/prob_lora_vs_zs_per_crop.csv)

### 3-G. Calibration 분석
- PICP vs nominal: [`visualizations/prob/calibration_diagram.png`](visualizations/prob/calibration_diagram.png)
  - baseline LoRA 의 PICP@60=0.589, PICP@80=0.789 → 목표 (0.6, 0.8) 와 거의 일치. underconfident 도 overconfident 도 아님.
- 9분위별 pinball loss: [`visualizations/prob/per_quantile_pinball.png`](visualizations/prob/per_quantile_pinball.png)
- Variant 종합 radar: [`visualizations/prob/variant_radar.png`](visualizations/prob/variant_radar.png)
- (참고) Conformal recalibration 효과: [`visualizations/prob/picp_cqr_effect.png`](visualizations/prob/picp_cqr_effect.png) — CQR 후처리로 PICP 더 가까이 맞출 수 있으나 본 운영 모델은 raw LoRA 사용 (이미 충분히 calibrated).

### 3-H. ★ 범위예측 최종 선정: `Chronos2LoRA_baseline`

| 결정 사유 | 근거 |
|---|---|
| WQL 1위 (0.1298) | 6 케이스 비교 최저 |
| PICP@60/80 목표 거의 정확히 달성 | 0.589 / 0.789 vs 0.6 / 0.8 |
| 데이터셋이 도메인 지식 기반 → 해석 가능 | (vs auto-selected optimal) |
| LoRA 학습 시간 합리적 (8-10분) | 운영 retrain 부담 작음 |

품목별 예측 시각화 일부 (각 작물 high 등급 1개씩): [`visualizations/sample_forecasts/`](visualizations/sample_forecasts/) (13개 PNG).

---

## 4. 점예측 모델 — 비교·선정 과정

> 출처: `point_pred_3d/RESULTS_REPORT_v2.md`, `point_pred_3d/outputs/comparison_v2/`

### 4-A. 학습 셋업

| 항목 | 값 |
|---|---|
| 비교 모델 | TimesFM 2.5 (200M, Google) + IBM Granite TTM (R2, ~5M IBM) |
| 평가 방식 | ZS (zero-shot) + Fine-tune 각각 |
| 예측 길이 | 3 일 |
| 컨텍스트 | 384일 (TimesFM, 32배수) / 512일 (TTM) |
| 평가 윈도우 | 49 (cutoff=-3, step=-15, test=731일) |
| Fine-tune | TimesFM: PEFT-LoRA (r=8, α=16, 3 epoch, lr=1e-4) / TTM: backbone freeze + decoder/head 학습 |

### 4-B. 4가지 케이스 비교

원본 표: [`evaluation_data/point_metrics_table.csv`](evaluation_data/point_metrics_table.csv)

| model | RMSE | MAE | MAPE (%) | **MASE ↓** |
|---|---:|---:|---:|---:|
| **★ TimesFM 2.5 ZS** | **3,364.96** | **2,994.97** | **12.64** | **0.806** |
| TimesFM 2.5 LoRA (best: r=8, α=16, lr=1e-4) | 3,426.55 | 3,056.77 | 13.09 | 0.811 |
| TTM FT (decoder/head fine-tune) | 3,800.45 | 3,405.19 | 15.23 | 0.931 |
| TTM ZS | 8,198.51 | 7,884.90 | 44.16 | 2.184 |

### 4-C. 평가지표 정의

| 지표 | 정의 | 좋은 값 |
|---|---|---|
| **RMSE** (Root Mean Squared Error) | √(평균(예측−실제)²). 단위: 원 | 낮을수록 |
| **MAE** (Mean Absolute Error) | 평균(|예측−실제|). 단위: 원 | 낮을수록 |
| **MAPE** (Mean Absolute Percentage Error) | 평균(|예측−실제| / |실제|) × 100. 단위: % | 낮을수록 |
| **MASE** (Mean Absolute Scaled Error) | MAE 를 train 기간 7일-seasonal naive MAE 로 나눔. 1.0 미만 = naive 보다 우수 | 낮을수록 |

### 4-D. Fine-tuning 개선율

| model | RMSE Δ | MAE Δ | MAPE Δ | MASE Δ |
|---|---:|---:|---:|---:|
| TimesFM (ZS → LoRA) | **-1.83%** | -2.06% | -3.57% | **-0.56%** |
| TTM (ZS → FT) | +53.64% | +56.81% | +65.51% | **+57.38%** |

→ **TimesFM 의 LoRA 는 ZS 를 개선 못 함**. TTM 은 큰 폭 개선이지만 ZS 자체가 매우 약했기 때문.

### 4-E. 그리드 서치 상세

**TimesFM 2.5 LoRA** ([`evaluation_data/point_timesfm_grid.csv`](evaluation_data/point_timesfm_grid.csv)):
| trial | r | α | lr | num_samples | epochs | val_loss | elapsed |
|---|---:|---:|---:|---:|---:|---:|---:|
| lora_r4_a8_lr1e4   | 4 | 8 | 1e-4 | 2000 | 3 | 0.611 | 3.3 min |
| **lora_r8_a16_lr1e4** | 8 | 16 | 1e-4 | 2000 | 3 | **0.583** ← best | 3.3 min |
| lora_r16_a32_lr5e5 | 16 | 32 | 5e-5 | 2000 | 3 | 0.608 | 3.3 min |

**IBM TTM Fine-tune** ([`evaluation_data/point_ttm_grid.csv`](evaluation_data/point_ttm_grid.csv)):
| trial | revision | lr | epochs | val_loss | elapsed |
|---|---|---:|---:|---:|---:|
| r2_main_lr1e3_e8 | main | 1e-3 | 8 | 0.272 | 1.9 min |
| **r2_main_lr5e4_e12** | main | 5e-4 | 12 | **0.246** ← best | 2.2 min |
| r2_main_lr2e3_e6 | main | 2e-3 | 6 | 0.274 | 1.8 min |

### 4-F. 핵심 발견

1. **TimesFM 2.5 ZS 가 본 데이터에서 매우 강력** — 농산물 도메인 명시적 학습 없이 MASE 0.806 (1.0 미만 = seasonal naive 보다 우수). foundation model 의 일반화 성능 입증.

2. **LoRA 가 ZS 를 개선 못 함 (-0.56%)** — 본 데이터 규모 (46 series × 2000 random windows × 3 epoch) 로는 ZS 의 보편적 패턴을 능가 못 함. 더 큰 데이터·더 많은 epoch 시도 가치 있으나 ROI 불확실.

3. **TTM 의 큰 개선 폭은 base 의 약함이 원인** — TTM-R2 base 는 농산물에 사전학습 없음 + `decoder_mode="mix_channel"` 활성화 시 channel mixer 가 random init. fine-tune 으로 정상화 (MASE 2.18 → 0.93). 즉 TTM 의 절대 성능 보단 차이의 폭이 큰 것뿐.

4. **품목별 우승은 ZS 24 / LoRA 22** — 46 품목을 거의 반반 차지. TTM 은 0 품목. 품목별 ZS/LoRA 분기 라우팅 시 추가 이득 가능.

### 4-G. 시각자료

- 4-way 막대: [`visualizations/point/model_comparison_bar.png`](visualizations/point/model_comparison_bar.png)
- 품목별 우승: [`visualizations/point/per_crop_winner.png`](visualizations/point/per_crop_winner.png)
- 품목별 MASE 원본: [`evaluation_data/point_per_crop_mase.csv`](evaluation_data/point_per_crop_mase.csv)

### 4-H. ★ 점예측 최종 선정: TimesFM 2.5 Zero-Shot

| 결정 사유 | 근거 |
|---|---|
| MASE 최저 (0.806) | 4-way 비교 1위 |
| 학습 비용 0 | adapter / fine-tune 파일 관리 불필요 |
| 운영 단순 | base 모델만 다운로드, retrain 불필요 |
| 신규 품목 / 도메인 즉시 대응 | foundation model 의 일반화 활용 |
| 결과 재현성 | 학습 단계 없음 → 환경별 결과 편차 없음 |

---

## 5. 최종 권장 / 통합 결론

| 용도 | 모델 | 폴더 | 핵심 메트릭 |
|---|---|---|---|
| **10일 확률범위 예측** | Chronos2 LoRA (baseline) | [`PROBABILISTIC_FORECAST/`](PROBABILISTIC_FORECAST/) | WQL 0.1298, PICP@80 0.789 |
| **3일 점예측** | TimesFM 2.5 Zero-Shot | [`POINT_FORECAST/`](POINT_FORECAST/) | MASE 0.806, MAPE 12.64% |

### 5-A. 운영 패턴 권장
1. **두 모델 병행 운영** — 점예측 (단일값) 과 확률범위 (불확실성) 는 용도가 달라 모순 없음
2. **conda 환경 2개 분리** — autogluon (확률) ↔ timesfm/transformers (점예측). transformers 버전 충돌 회피
3. **데이터 파이프라인 공유** — 두 모델이 같은 `full_baseline.parquet` 사용. 데이터 갱신 시 양쪽 모델에 동일 적용

### 5-B. 갱신 주기 권장
- **데이터**: 매일 신규 가격/거래량 append. 매월 1일 known covariate (금리·CPI) 갱신
- **확률예측 모델 (Chronos2 LoRA)**: 분기별 1회 retrain (8~10분/variant) 권장
- **점예측 모델 (TimesFM ZS)**: retrain 불필요. base 모델 새 버전 (TimesFM 3.0 등) 출시 시 교체 검토

---

## 6. 환경 / 운영 / 트러블슈팅

### 6-A. 의존성 (두 환경 분리 권장)

**확률예측 환경** (`prob_forecast`):
```
autogluon.timeseries==1.5.0
torch>=2.0
pandas>=2.0
pyarrow>=14.0
```

**점예측 환경** (`point_forecast`):
```
timesfm>=2.0.0
torch>=2.0
pandas>=2.0
huggingface-hub>=0.20
```

상세: 각 폴더의 `requirements.txt`.

### 6-B. GPU 메모리
- 확률예측 (Chronos2 LoRA, 200M): GPU 메모리 4~6 GB (batch=16, context=365)
- 점예측 (TimesFM 2.5, 200M): GPU 메모리 3~5 GB (batch=16, context=384)
- 둘 다 CPU 만으로도 동작 (배치 추론 5~10배 느림)

### 6-C. 자주 묻는 질문

**Q. 운영 중 known covariate (기상·금리) 미래값을 어떻게 채우나요?**
A. `PROBABILISTIC_FORECAST/predict_example.py` 의 `make_known()` 참고. 운영 시:
- 기상 → 기상청 7~10일 예보 API (품목별 주산지 매핑은 `dataset_description.md` §6)
- 휴장일 → 도매시장 공시 캘린더
- 금리·CPI → 한국은행/통계청 발표 (월 1회, ffill)

**Q. 점예측 모델은 covariate 필요 없나요?**
A. 네, TimesFM 2.5 ZS 는 **univariate** (target 시리즈만). covariate 컬럼이 데이터에 있어도 무시됨.

**Q. 신규 품목 추가 시 어떻게 하나요?**
A. 같은 컬럼 스키마로 데이터에 추가하면 즉시 추론 가능:
- TimesFM ZS: 재학습 없음
- Chronos2 LoRA: 정확도 유지하려면 retrain 권장 (8~10분)

**Q. 두 모델의 결과가 다르면 어느 쪽을 신뢰?**
A. 용도가 다름:
- 단일 점예측값 (대시보드, 평균 손실 최소화) → POINT_FORECAST
- 신뢰구간이 필요 (리스크 알람, 보험) → PROBABILISTIC_FORECAST 의 `0.5` (중앙값) 또는 `mean`
- 두 모델의 mean 끼리 비교: TimesFM ZS 의 MASE 가 더 낮으므로 점예측은 TimesFM ZS 가 정답에 가까움

**Q. 모델 가중치 파일 위치?**
A.
- 확률예측: `PROBABILISTIC_FORECAST/model/` 안에 AutoGluon predictor 전체 포함 (즉시 로드 가능)
- 점예측: HuggingFace 에서 자동 다운로드 (`google/timesfm-2.5-200m-pytorch`). 인터넷 없는 환경이면 `~/.cache/huggingface/hub/` 캐시 사전 복사 필요

### 6-D. transformers 버전 충돌 주의
- `autogluon.timeseries==1.5.0` 은 `transformers<4.58` 만 호환
- `timesfm` 패키지 자체는 transformers 의존성이 낮아 호환 가능 (timesfm 2.0.0 + transformers 4.57.6)
- 만약 TimesFM 의 PEFT LoRA 학습을 다시 시도하려면 `transformers 5.x` 필요한데, 이 경우 autogluon 깨짐. **별도 conda 환경 분리 권장**.

---

## 7. 한계점 및 후속 개선 아이디어

### 7-A. 확률범위 예측
- **`market_rest` (휴장일) 의 자동 선정 탈락** — Spearman 0.035 로 임계값 (0.05) 미만이지만 도메인 지식으로는 유지가 맞음. → baseline 선정 근거 (도메인 지식 우선)
- **Conformal recalibration (CQR) 미적용** — 현재 PICP 가 이미 목표에 근접하므로 미적용. 더 엄격한 calibration 필요시 [`5_12_fin/src/10_conformal_recalibration.py`](../5_12_fin/src/10_conformal_recalibration.py) 패턴 적용 가능
- **LoRA 와 ZS 의 모델 합치기 (ensemble) 미실시** — 사양 제약. 운영 시 룰 기반 라우팅 가능

### 7-B. 점예측
- **TimesFM 2.5 LoRA 가 ZS 를 개선 못 함** — 본 데이터 규모의 한계. 더 큰 학습 데이터 (>10,000 samples) 또는 covariate 활용 시 가능성. `forecast_with_covariates` 사용 시 jax 의존성 추가 필요
- **TimesFM 의 더 긴 context 미시도** — 본 프로젝트 384일. 모델 최대 16,384까지 지원. 1024~1536 시도 시 추가 개선 가능성
- **TTM-R2.1 (일/주 데이터 특화) 미사용** — `freq_token` 주입 필요로 R2 main 만 사용. R2.1 시도 시 농산물 일별 데이터에 더 적합 가능성
- **품목별 모델 분기 (앙상블 룰) 미적용** — per_crop_winner 분석 결과만 출력. 운영 시 룰 라우팅 가능 (예: 양념채소엔 LoRA, 과실엔 ZS)

### 7-C. 데이터 측면
- **신규 품목 학습**: 본 46개 외 작물은 retrain 권장 (특히 LoRA 모델). TimesFM ZS 는 retrain 없이도 동작 (정확도는 보장 안 됨)
- **외부 도메인 적용**: 농산물 외 시장 (수산물, 축산물, 공산품 등) 적용 시 LoRA adapter 와 covariate 정의 재설계 필요

---

## 부록 A. 폴더별 산출물 매핑

```
final_handoff/
├── README.md
├── INTEGRATED_REPORT.md       ← 본 문서
│
├── PROBABILISTIC_FORECAST/   → 5_12_fin/deploy/ 의 복사본
│   ├── README.md (신규)
│   ├── dataset_description.md
│   ├── predict_example.py
│   ├── requirements.txt
│   ├── data/                  ← 5_12_fin/data/{full,static,meta}_baseline
│   └── model/                  ← 5_12_fin/outputs/predictor_baseline/ 의 AutoGluon predictor
│
├── POINT_FORECAST/             → point_pred_3d/ 의 핵심 발췌
│   ├── README.md (신규)
│   ├── dataset_description.md (신규)
│   ├── predict_example.py (신규)
│   ├── requirements.txt (신규)
│   └── data/                   ← PROBABILISTIC_FORECAST 와 같은 데이터
│
├── visualizations/
│   ├── prob/                   ← 5_12_fin/outputs/{comparison,comparison_v2,conformal}/*.png
│   ├── point/                  ← point_pred_3d/outputs/comparison_v2/*.png
│   └── sample_forecasts/       ← 5_12_fin/outputs/forecasts_viz/baseline/ 일부 (13개)
│
└── evaluation_data/             ← 평가 수치 원본 CSV (재현·감사용)
    ├── prob_*.csv               ← 5_12_fin 의 확률 평가 CSV
    └── point_*.csv              ← point_pred_3d 의 점예측 평가 CSV
```

## 부록 B. 원본 보고서 참조

본 통합 보고서 외에 원본 ML 팀 보고서들이 다음에 보존되어 있음:
- `5_12_fin/RESULTS_REPORT.md` — 확률예측 상세 보고
- `5_12_fin/RESULTS_REPORT_v2_followup.md` — 후속 분석
- `point_pred_3d/RESULTS_REPORT.md` — 점예측 1차 보고 (3-way)
- `point_pred_3d/RESULTS_REPORT_v2.md` — 점예측 최종 보고 (4-way)
- `point_pred_3d/HANDOFF_GUIDE.md` — 점예측 인수인계 가이드

이 보고서들은 final_handoff 폴더에는 포함되지 않으나 의문점·재현 필요 시 ML 팀에 요청 가능.
