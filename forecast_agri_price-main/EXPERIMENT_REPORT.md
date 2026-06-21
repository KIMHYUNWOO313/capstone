# EXPERIMENT_REPORT — 농산물 가격 예측 모델 종합 실험 보고서 (발표용)

**작성일**: 2026-05-18
**대상**: 최종 발표, ML 검토, 인계 후 백엔드 이해

본 보고서는 **두 최종 모델** (Chronos2 LoRA 10일 확률예측 · TimesFM 2.5 ZS 3일 점예측) 의 선정 과정에서 수행한 **모든 실험 결과·비교·통계치**를 한 곳에 모아 발표 자료로 즉시 활용할 수 있도록 작성된 자료입니다. `00_INTEGRATED_REPORT.md` 가 narrative 라면 본 문서는 **표·그래프 중심의 정량 자료** 입니다.

---

## 목차

1. [연구 개요와 평가 셋업](#1-연구-개요와-평가-셋업)
2. [데이터 파이프라인 실험 — 3 variant 비교](#2-데이터-파이프라인-실험--3-variant-비교)
3. [점예측 모델 실험 — TimesFM / TTM / xreg](#3-점예측-모델-실험)
4. [구간예측 모델 실험 — Chronos2 ZS vs LoRA × 3 variant](#4-구간예측-모델-실험)
5. [일기예보 통합 실험 — 운영 적용성 검증](#5-일기예보-통합-실험)
6. [최종 모델 선정 정당화](#6-최종-모델-선정-정당화)
7. [한계와 향후 과제](#7-한계와-향후-과제)
8. [부록 — 원본 데이터 소재 매핑](#8-부록--원본-데이터-소재-매핑)

---

## 1. 연구 개요와 평가 셋업

### 1-1. 연구 문제

**46개 농산물 품목** (13 작물 × 1~4 등급) 의 도매가격을 다음 두 가지 형태로 예측:

| 용도 | 예측 | 출력 |
|---|---|---|
| 점예측 | 3일 후 가격 (mean) | scalar/일 |
| 구간예측 | 10일 후 가격 (9분위 + mean) | 80% / 60% 신뢰구간 |

### 1-2. 데이터

- **기간**: 2015-11-16 ~ 2026-02-28 (일별, ~10년)
- **품목 수**: 46
- **Train/Test**: train_end = 2024-02-29, test = 2024-03-01 ~ 2026-02-28 (731일)

### 1-3. 평가 방식 (롤링 윈도우)

| 모델 | Horizon | Cutoff step | 윈도우 수 | 총 평가 수 |
|---|---:|---:|---:|---:|
| 점예측 (TimesFM/TTM) | 3일 | 15일 | 49 | 49 × 46 × 3 = **6,762** |
| 구간예측 (Chronos2) | 10일 | 15일 | 49 | 49 × 46 × 10 = **22,540** |

### 1-4. 환경

- 하드웨어: NVIDIA RTX 5060 Ti 16 GB
- 환경: anaconda capstone (autogluon.timeseries==1.5.0, timesfm 2.0.0, jax 0.9.2, tsfm_public 0.3.5)

---

## 2. 데이터 파이프라인 실험 — 3 variant 비교

### 2-1. 5 도메인 원천 데이터

| 도메인 | 출처 | 컬럼 | 주기 |
|---|---|---|---|
| 농산물 가격 | aT 도매시장 | 도매 평균가(원), 거래량 | 일별 |
| 기상 | 기상청 지상관측 (13 주산지) | 일평균 기온, 일교차, 강수, 풍속, 습도, 기압, 일조 | 일별 |
| 거시경제 | 한국은행·통계청 | 기준금리, CPI, M2, 국고채 3년 | 월별 (ffill) |
| 에너지 | 한국석유공사 | 면세 경유가 | 일별 |
| 뉴스 감성 | 자체 크롤링 | 농업 감성지수 (-1~1) | 일별 |

### 2-2. 전처리 6단계

```
① item_id 생성 (46품목)
② 품목 → 주산지 매핑 (Spearman 상관 기반, item_station_map)
③ 결측 처리: target/amount 선형보간, 기타 ffill
④ known/past 분류 정정 (oil_tax_free_diesel: known→past 정정, 매일 변동)
⑤ Train/Test 누설 검증 (v9_1 보간 누설 버그 차단)
⑥ Parquet 저장 (MultiIndex item_id × timestamp)
```

### 2-3. 3 variant 정의

| variant | rows | cols | 구성 의도 | Known | Past |
|---|---:|---:|---|---:|---:|
| **baseline** | 172,868 | 13 | 도메인 지식 기반 | 5 | 7 |
| no_weather | 172,868 | 7 | 기상 컬럼 모두 제거 (간소화 가설) | 3 | 3 |
| optimal | 172,868 | 15 | Spearman + SHAP + VIF 자동 선정 | 4 | 10 |

### 2-4. 구간예측 (Chronos2) 에서의 variant 비교 결과

| variant | model | **WQL ↓** | CRPS ↓ | PICP@60 | PICP@80 | MSIS@80 | Sharpness@80 |
|---|---|---:|---:|---:|---:|---:|---:|
| **★ baseline** | **LoRA** | **0.1298** | 3,195 | **0.589** ✅ | **0.789** ✅ | 5.22 | 12,187 |
| baseline | ZS | 0.1369 | 3,361 | 0.573 | 0.783 | 5.48 | 12,788 |
| no_weather | LoRA | 0.1301 | 3,193 | 0.572 | 0.772 | 5.21 | 12,095 |
| no_weather | ZS | 0.1384 | 3,404 | 0.570 | 0.780 | 5.51 | 12,984 |
| optimal | LoRA | 0.1326 | 3,270 | 0.567 | 0.774 | 5.32 | 12,271 |
| optimal | ZS | 0.1397 | 3,431 | 0.552 | 0.768 | 5.57 | 13,016 |

### 2-5. 핵심 발견

1. **baseline LoRA**: WQL 0.1298 (1위), PICP@60 0.589 vs 목표 0.6, PICP@80 0.789 vs 목표 0.8 → **거의 완벽한 calibration**
2. **no_weather LoRA**: WQL 0.1301 (+0.2%) → 기상 변수 기여도 미미. 단순화 관점에선 동등 후보
3. **optimal LoRA**: WQL 0.1326 → 자동 선정이 노이즈 키움. 도메인 지식 우위 사례
4. **LoRA vs ZS** (variant 3종 평균): WQL 5~6% 일관 개선

---

## 3. 점예측 모델 실험

### 3-1. 4-way 기본 비교 (TimesFM × TTM × ZS/FT)

원본: `03_POINT_FORECAST/experiments/metrics_table_4way.csv`

| model | RMSE | MAE | MAPE (%) | **MASE ↓** |
|---|---:|---:|---:|---:|
| **★ TimesFM 2.5 ZS** | **3,364.96** | **2,994.97** | **12.64** | **0.806** |
| TimesFM 2.5 LoRA (r=8, α=16, lr=1e-4) | 3,426.55 | 3,056.77 | 13.09 | 0.811 |
| TTM-R2 FT (decoder/head, dyn 12) | 3,800.45 | 3,405.19 | 15.23 | 0.931 |
| TTM-R2 ZS | 8,198.51 | 7,884.90 | 44.16 | 2.184 |

### 3-2. LoRA 그리드 서치 (TimesFM 2.5)

원본: `03_POINT_FORECAST/experiments/grid_timesfm.csv`

| trial | r | α | lr | epochs | val_loss | elapsed |
|---|---:|---:|---:|---:|---:|---:|
| lora_r4_a8_lr1e4 | 4 | 8 | 1e-4 | 3 | 0.611 | 3.3 min |
| **lora_r8_a16_lr1e4** ★ | **8** | **16** | **1e-4** | **3** | **0.583** | 3.3 min |
| lora_r16_a32_lr5e5 | 16 | 32 | 5e-5 | 3 | 0.608 | 3.3 min |

### 3-3. 7-way Ablation — 다변량 활용 5가지 시도 + 단변량 ZS/LoRA

원본: `03_POINT_FORECAST/experiments/metrics_table_7way.csv`

| 순위 | model | RMSE | MAE | MAPE | **MASE ↓** | ZS 대비 |
|---:|---|---:|---:|---:|---:|---:|
| 1 | **★ TimesFM 2.5 ZS (단변량)** | **3,365** | **2,995** | **12.64** | **0.806** | — |
| 2 | TimesFM 2.5 LoRA (단변량 fine-tune) | 3,427 | 3,057 | 13.09 | 0.811 | +0.6% |
| 3 | TimesFM xreg + timesfm (다변량, dyn 5) | 3,551 | 3,175 | 13.60 | 0.862 | **+6.9%** |
| 4 | TimesFM xreg + static (dyn 5 + crop/grade) | 3,563 | 3,185 | 13.66 | 0.865 | +7.3% |
| 5 | IBM TTM FT (다변량, dyn 12 + channel mixing) | 3,800 | 3,405 | 15.23 | 0.931 | +15.4% |
| 6 | TimesFM timesfm + xreg (잔차 보정) | 4,150 | 3,795 | 17.38 | 1.029 | +27.6% |
| 7 | IBM TTM ZS | 8,199 | 7,885 | 44.16 | 2.184 | +170% |

#### 품목별 우승 (46품목)

| winner | 품목 수 | 비율 |
|---|---:|---:|
| TimesFM_ZS | 22 | 48% |
| TimesFM_LoRA | 21 | 46% |
| TimesFM_xreg_a | 1 | 2% |
| TimesFM_xreg_b | 1 | 2% |
| TimesFM_xreg_static | 1 | 2% |
| TTM_FT / TTM_ZS | 0 / 0 | 0% |

**단변량 ZS + LoRA = 43/46 = 93%**.

### 3-4. ★ NEW — 49-window 백테스트 (ZS vs xreg_oracle vs xreg_noisy)

> **목적**: 운영에서 미래 일기예보를 dynamic_numerical_covariate(xreg) 로 주입하는 가치를 정량 평가.
> - **xreg_oracle**: 미래 weather = 실측값 (일기예보 100% 정확 가정, xreg 메커니즘의 이론적 상한)
> - **xreg_noisy**: 미래 weather = 실측 + 가우시안 노이즈 (기상청 단기예보 실측 오차 시뮬레이션)
>   - 기온 ±1.5℃, 습도 ±5%, 강수확률 ±20%, 풍속 ±0.5m/s
>
> 원본: `tmp/tmp_timesf2.5/REPORT.md`, `outputs/metrics_overall.csv`

#### 전체 결과 (6,762 예측)

| 모드 | MAE (₩) | MAPE (%) | MASE | vs ZS |
|---|---:|---:|---:|---:|
| **zero-shot (baseline)** | **3,289** | **12.85** | **1.8656** | — |
| xreg + oracle | 3,410 | 13.60 | 1.9504 | **+4.5%** |
| xreg + noisy | 3,413 | 13.62 | 1.9554 | **+4.8%** |

#### Horizon 별

| Horizon | ZS MASE | oracle MASE | noisy MASE | noisy vs zs |
|---|---:|---:|---:|---:|
| D+1 | 1.3383 | 1.4356 | 1.4430 | +7.8% |
| D+2 | 1.9059 | 1.9829 | 1.9887 | +4.3% |
| D+3 | 2.3526 | 2.4327 | 2.4346 | +3.5% |

#### 품목별

- 46 품목 중 **xreg 가 ZS 보다 좋은 품목은 단 2개** (`cucumber_bdadagi_ea100_mid`, `sweetpotato_box10kg_premium`, 차이 < 0.01 MASE)
- 가장 큰 악화: `spinach_box4kg_mid/high` +0.30 MASE, `cucumber_bdadagi_ea100_low` +0.25 MASE
- 결론: **xreg 미적용 결정. 일기예보 활용은 점예측에 의미 없음.**

#### 작물별

| 작물 | ZS MASE | xreg_noisy MASE | Δ |
|---|---:|---:|---:|
| apple | 2.5696 | 2.5979 | +1.1% |
| cabbage | 1.7545 | 1.8121 | +3.3% |
| carrot | 1.9566 | 2.1195 | +8.3% |
| crown_daisy | 1.7073 | 1.7370 | +1.7% |
| cucumber_bdadagi | 2.2027 | 2.2762 | +3.3% |
| garlic_chive | 1.4818 | 1.5671 | +5.8% |
| honewort | 2.4417 | 2.5518 | +4.5% |
| napa_cabbage | 1.8784 | 1.9497 | +3.8% |
| onion | 1.5743 | 1.7110 | +8.7% |
| perilla_leaf | 1.8283 | 1.9100 | +4.5% |
| potato_sumi | 1.7179 | 1.8205 | +6.0% |
| spinach | 2.1756 | 2.4292 | +11.7% |
| sweetpotato | 1.0637 | 1.0843 | +1.9% |

→ **13개 작물 전부 ZS 가 우세**.

### 3-5. ★ NEW — TTM 4-covariate Fair 비교

> **배경**: 4-way 기본 비교의 TTM FT 는 dyn 12 개 (past 7 포함) 를 사용 → "운영 시 실제로 활용 가능한 정보" 가 아니므로 불공평. **운영에서 known-future 로 확보 가능한 단기예보 4종 만으로 fine-tune** 하여 공평하게 재평가.
>
> Control Covariates: `weather_temp_range`, `weather_humidity_avg`, `precip_prob`, `weather_wind_avg`
> 원본: `tmp/tmp_ttm/REPORT.md`, `outputs/metrics_overall.csv`

#### 전체 결과 (6,762 예측)

| 모델 | MAE (₩) | MAPE (%) | MASE | vs ZS |
|---|---:|---:|---:|---:|
| **TimesFM 2.5 ZS** | **2,995** | **12.65** | **1.7355** | — |
| TTM-R2 FT (4 cov, 806K params, mix_channel) | 3,200 | 13.41 | 1.8450 | **+6.3%** |

#### Horizon 별

| Horizon | MAE_zs | MAE_ttm | MASE_zs | MASE_ttm | TTM vs ZS |
|---|---:|---:|---:|---:|---:|
| D+1 | 2,249 | 2,494 | 1.25 | 1.40 | **+10.9%** |
| D+2 | 3,138 | 3,346 | 1.80 | 1.90 | +6.6% |
| D+3 | 3,599 | 3,762 | 2.15 | 2.24 | +4.5% |

#### 품목별 (top/bottom 5)

| 품목 | MASE_zs | MASE_ttm | Δ |
|---|---:|---:|---:|
| **개선 top:** honewort_kg4_low | 2.3006 | 2.1727 | -5.56% |
| crown_daisy_box4kg_low | 1.9590 | 1.8956 | -3.24% |
| honewort_kg4_mid | 2.4486 | 2.3692 | -3.24% |
| cucumber_bdadagi_ea100_mid | 1.9284 | 1.8712 | -2.97% |
| cabbage_net8kg_high | 1.6287 | 1.5991 | -1.82% |
| **악화 top:** carrot_box20kg_mid | 1.4282 | 1.9135 | **+33.98%** |
| carrot_box20kg_high | 1.7518 | 2.2622 | **+29.14%** |
| cucumber_bdadagi_ea100_premium | 1.5685 | 1.8891 | +20.44% |
| perilla_leaf_bunch100_high | 1.4795 | 1.7553 | +18.64% |
| potato_sumi_box20kg_high | 1.0363 | 1.2219 | +17.91% |

**TTM 개선 10 품목 / 악화 36 품목 (78% ZS 우세)**.

#### TTM 학습 메타데이터

| 항목 | 값 |
|---|---|
| Base model | `ibm-granite/granite-timeseries-ttm-r2` (806K params) |
| context_length / prediction_length / prediction_filter | 512 / 96 / 3 |
| decoder_mode | `mix_channel` (covariate fusion 핵심) |
| lr / epochs / batch | 5e-4 / 8 (early stopping at 8) / 32 |
| Train / Val / Test | 2015-11-16~2023-08-31 / 2023-09-01~2024-02-29 / 2024-03-01~2026-02-28 |
| 학습 시간 | 941.6 초 (~16분, RTX 5060 Ti) |
| Final eval_loss | 0.383 |

#### 결론

- 4 known-future covariate 만으로도 **TimesFM 2.5 ZS 우위 (+6.3%)**
- 기존 12 cov TTM FT (MASE 0.931, +15.4%) 보다는 5.9% 개선 — past_covariate 가 noise 로 작용했을 가능성
- 그러나 ZS 를 못 넘는다는 **결론 자체는 동일** → 데이터 규모가 아니라 모델 일반화 능력 차이
- 운영 도입 비용 (매 회 fine-tune ~16분 필요) 대비 ROI 없음 → **TTM 도입 불가**

### 3-6. 다변량이 ZS 를 능가 못 한 이유 (인과 분석)

1. **TimesFM 2.5 의 사전학습 일반화가 매우 강력** — 200M params 가 다양한 도메인 시계열로 사전학습. 농산물 가격 패턴 (계절성·수준·변동성) 내부에 weather/macro signal 이 이미 흡수되어 있음.
2. **xreg = ridge regression 의 선형 한계** — covariate 의 선형 관계만 모델링. 농산물 가격↔weather 의 **비선형·시차·교호작용** 못 잡음.
3. **잔차에 추가 signal 거의 없음** — TimesFM ZS 가 이미 대부분 variance 설명. 잔차는 거의 white noise → ridge 가 노이즈에 overfit.
4. **TTM (806K) vs TimesFM (200M) 의 사전학습 격차** — 파라미터 규모 + 사전학습 데이터 격차가 그대로 성능 격차로 반영.
5. **xreg + timesfm vs timesfm + xreg 비대칭** (0.862 vs 1.029, 19% 차이) — covariate 가 noise 만은 아니라는 정량 증거 (만약 noise 만이면 두 모드 결과가 동등하게 악화되어야 함).

### 3-7. ★ 데이터 quality 의 정량적 반증 (3가지 증거)

#### 증거 1: TTM FT 의 +57.4% 큰 폭 개선

| | TTM ZS | TTM FT | Δ |
|---|---:|---:|---:|
| MASE | 2.184 | 0.931 | **-57.4%** |
| RMSE | 8,199 | 3,800 | -53.6% |
| MAE | 7,885 | 3,405 | -56.8% |

만약 데이터가 노이즈투성이라면 random init channel mixer 의 fine-tune 이 의미 있는 representation 학습 불가. **데이터에 학습 가능한 covariate-target 신호가 풍부함을 직접 증명**.

#### 증거 2: Chronos2 LoRA 의 일관된 5~6% 개선

| variant | WQL 개선 (ZS→LoRA) | CRPS 개선 |
|---|---:|---:|
| baseline | -5.2% | -4.9% |
| no_weather | -6.0% | -6.2% |
| optimal | -5.1% | -4.7% |

데이터 구성을 바꿔도 일관 개선. PICP 가 목표값 (0.6, 0.8) 에 거의 정확히 수렴 = 데이터의 신호·불확실성이 적절히 담겨 있음의 증거.

#### 증거 3: xreg 두 모드 비대칭

| 모드 | MASE | 해석 |
|---|---:|---|
| xreg + timesfm | 0.862 | covariate 회귀 먼저 → 잔차에 TimesFM |
| timesfm + xreg | 1.029 | TimesFM → 잔차에 covariate 회귀 |

19% 차이 = covariate 가 target 의 일정 비중을 **선형적으로 설명**한다는 정량 증거.

---

## 4. 구간예측 모델 실험

### 4-1. 6-way 케이스 비교 (variant 3 × {ZS, LoRA})

§2-4 와 동일. 핵심 표 재인용:

| variant | model | **WQL ↓** | CRPS ↓ | PICP@60 | PICP@80 | MSIS@80 | Sharpness@80 |
|---|---|---:|---:|---:|---:|---:|---:|
| **★ baseline** | **LoRA** | **0.1298** | 3,195 | 0.589 ✅ | 0.789 ✅ | 5.22 | 12,187 |
| baseline | ZS | 0.1369 | 3,361 | 0.573 | 0.783 | 5.48 | 12,788 |
| no_weather | LoRA | 0.1301 | 3,193 | 0.572 | 0.772 | 5.21 | 12,095 |
| no_weather | ZS | 0.1384 | 3,404 | 0.570 | 0.780 | 5.51 | 12,984 |
| optimal | LoRA | 0.1326 | 3,270 | 0.567 | 0.774 | 5.32 | 12,271 |
| optimal | ZS | 0.1397 | 3,431 | 0.552 | 0.768 | 5.57 | 13,016 |

### 4-2. Calibration 분석

**baseline LoRA** 의 PICP:
- @60 (목표 0.6): **0.589** — underestimate 1.1pp, 거의 perfect
- @80 (목표 0.8): **0.789** — underestimate 1.1pp, 거의 perfect
- → **운영 신뢰구간으로 즉시 활용 가능**. Conformal recalibration 불필요.

### 4-3. LoRA 학습 설정

| 항목 | 값 |
|---|---|
| Base | `autogluon/chronos-2` (200M params) |
| LoRA | r=16, α=32 |
| Steps | 4000 |
| Batch | 16 |
| Learning rate | 5e-5 |
| Context | 365일 |
| Horizon | 10일 |
| 학습 시간 | 8~10분 (RTX 5060 Ti), variant 당 |

### 4-4. LoRA vs ZS 작물별 개선 (baseline)

원본: `02_PROBABILISTIC_FORECAST/experiments/lora_vs_zs_per_crop.csv`

| 작물 | ZS WQL | LoRA WQL | Δ (LoRA-ZS) |
|---|---:|---:|---:|
| apple | 0.1041 | 0.0961 | -7.7% |
| cabbage | 0.2228 | 0.2106 | -5.5% |
| carrot | 0.1283 | 0.1240 | -3.4% |
| crown_daisy | 0.1644 | 0.1574 | -4.2% |
| cucumber_bdadagi | 0.1571 | 0.1486 | -5.4% |
| garlic_chive | 0.1052 | 0.1003 | -4.7% |
| honewort | 0.1602 | 0.1530 | -4.5% |
| napa_cabbage | 0.1797 | 0.1683 | -6.4% |
| onion | 0.1207 | 0.1153 | -4.5% |
| perilla_leaf | 0.1182 | 0.1146 | -3.0% |
| potato_sumi | 0.0973 | 0.0900 | -7.5% |
| spinach | 0.1466 | 0.1378 | -6.0% |
| sweetpotato | 0.0759 | 0.0726 | -4.3% |

→ 13개 작물 **전부 LoRA 개선**.

### 4-5. 평가 지표 정의

| 지표 | 정의 | 좋은 값 |
|---|---|---|
| WQL | Chronos 공식 손실. 9분위 pinball loss 가중합 / |y| | 낮을수록 |
| CRPS | 9분위 평균 pinball × 2. 적분형 CRPS 의 9분위 근사 | 낮을수록 |
| PICP@α | 신뢰구간 α 안 실제값 비율. 목표값에 근접해야 calibrated | 목표값(0.6/0.8) 근접 |
| MSIS@α (M4) | 신뢰구간 width + 페널티. seasonal naive MAE 로 스케일 | 낮을수록 |
| Sharpness@α | 신뢰구간 평균 width | 낮을수록 (calibration 유지하면서) |

---

## 5. 일기예보 통합 실험

### 5-1. 동기

- Chronos2 LoRA 의 `known_covariates` 5종 중 `weather_temp_range` 는 **미래 시점 값이 매일 변동**하는 변수
- 기존 `predict_example.py` 의 `make_known()` 은 **테스트셋의 실제 미래값 lookup** → 운영 불가
- 운영에서는 기상청 일기예보로 대체 필요

### 5-2. 기상청 두 Open API 활용

| API | Endpoint | Horizon | 발표시각 | 활용 |
|---|---|---|---|---|
| 단기예보 (`getVilageFcst`) | `VilageFcstInfoService_2.0` | D+0~D+3 | 02·05·08·11·14·17·20·23 | D+1~D+3 일교차 |
| 중기기온 (`getMidTa`) | `MidFcstInfoService` | D+4~D+10 | 06·18 | D+4~D+10 일교차 |

### 5-3. 18개 산지 매핑

품목 → 주산지(영문) → `(nx, ny)` 격자 (단기) + `regId` 시·군 구역 (중기). 자세한 매핑은 `05_WEATHER_FORECAST/stn_grid_map.py`.

### 5-4. ★ Chronos2 통합 검증 (apple_fuji_box10kg_high 1품목)

원본: `tmp/weather_xreg_pilot/REPORT.md`, `outputs/chronos2_forecast_with_weather.parquet`

| Horizon | Date | p10 | p50 | p90 | 80% 폭 |
|---|---|---:|---:|---:|---:|
| D+1 | 2026-05-16 | 41,443 | 50,045 | 60,377 | 18,934 |
| D+2 | 2026-05-17 | 41,267 | 51,459 | 63,360 | 22,093 |
| D+3 | 2026-05-18 | 42,633 | 54,453 | 69,019 | 26,386 |
| D+4 | 2026-05-19 | 42,138 | 53,873 | 71,247 | 29,109 |
| ... | ... | ... | ... | ... | ... |
| D+10 | 2026-05-25 | 40,245 | 54,781 | 74,869 | 34,624 |

- **운영 검증 통과**: 460행 (46품목 × 10일) 모두 정상 산출, NaN 없음
- 단기 D+1~D+3 일교차 + 중기 D+4~D+10 일교차가 자동으로 매핑됨

### 5-5. ★ TimesFM xreg 점예측 검증 (apple_fuji_box10kg_high)

원본: `tmp/weather_xreg_pilot/outputs/timesfm_forecast_xreg.parquet`

| Horizon | Date | ZS (₩) | xreg (₩) | Δ |
|---|---|---:|---:|---:|
| D+1 | 2026-05-16 | 50,492 | 51,479 | +1.95% |
| D+2 | 2026-05-17 | 55,016 | 56,640 | +2.95% |
| D+3 | 2026-05-18 | 58,789 | 61,857 | +5.21% |
| 평균 | | 54,766 | 56,659 | **+3.46%** |

단기예보 5/16~18 일교차 확대 + 강수확률 상승 신호를 ridge 가 반영해 가격 상향 예측. **단**, §3-4 의 49-window 백테스트에서는 xreg 가 ZS 대비 **MASE +4.8% 악화** 로 확인되었으므로 **운영 미적용**.

### 5-6. 점예측 vs 구간예측 — 일기예보 적용 결정 요약

| 모델 | 일기예보 | 근거 |
|---|---|---|
| **TimesFM 2.5 점예측 (3일)** | ❌ 미적용 | 49-window 백테스트: MASE +4.8% 악화, 13개 작물 전부 ZS 우세 |
| **Chronos2 LoRA 구간예측 (10일)** | ✅ 적용 | known_covariates 구조상 미래값 필수. 일기예보로 채워야 운영 가능 |

### 5-7. 모듈 구성 (`05_WEATHER_FORECAST/`)

```
05_WEATHER_FORECAST/
├── README.md                 ── 사용법, 산지 매핑 표, 운영 주의사항
├── requirements.txt          ── requests, python-dotenv, pandas, pyarrow
├── .env / .env.example       ── 단기/중기 API 키
├── stn_grid_map.py           ── 산지 18개 → (nx,ny) + regId
├── weather_fetcher.py        ── 단기/중기 API 호출 + 일별 집계
└── covariate_builder.py      ── 46품목 × 10일 known_covariates DataFrame
```

`build_known_covariates_frame()` 호출 한 줄로 운영 준비 완료. `02_PROBABILISTIC_FORECAST/predict_example.py` 의 `make_known()` 이 이를 호출.

---

## 6. 최종 모델 선정 정당화

### 6-1. 점예측: TimesFM 2.5 Zero-Shot (단변량)

| 정당화 항목 | 근거 |
|---|---|
| 모든 metric 최고 (4-way) | MASE 0.806, MAE 2,995, MAPE 12.64% |
| 7-way ablation 우위 | 다변량 5가지 시도 (xreg 3 + TTM 2) 모두 능가 (+6.9~170%) |
| 품목별 우승 우위 | 단변량 ZS + LoRA = 43/46 = 93% |
| 49-window 백테스트 우위 | xreg_oracle/noisy 모두 +4.5/4.8% 악화 |
| TTM 4-cov fair 비교 우위 | TTM FT +6.3% 악화, 36/46 품목 ZS 우세 |
| 학습 비용 0 | adapter 관리 불필요, 신규 품목 즉시 대응 |
| 재현성 | 학습 단계 없음 → 환경별 결과 편차 없음 |

### 6-2. 구간예측: Chronos2 LoRA + 일기예보 (baseline variant)

| 정당화 항목 | 근거 |
|---|---|
| 6-way 비교 1위 | WQL 0.1298 (다음 0.1301 보다 0.2% 우위) |
| 완벽한 calibration | PICP@60/80 = 0.589/0.789 ≈ 목표 0.6/0.8 (오차 1.1pp) |
| 일관된 LoRA 효과 | 3 variant 모두에서 ZS 대비 5~6% 개선 |
| 운영 가능성 | 일기예보 API 통합 완료 (05_WEATHER_FORECAST 모듈) |
| 학습 비용 합리적 | 8~10분/variant, 분기별 retrain 가능 |

### 6-3. "다변량을 안 썼다 → 다변량을 다 시도해봤다" 메시지

> 12 covariate (5 known + 7 past) + 4 정적 속성 의 다변량 데이터셋을 구성한 뒤, **TimesFM 의 공식 xreg 모듈 3 케이스** + **IBM TTM 의 fine-tune 2 케이스 (dyn 12, dyn 4)** 로 **총 5가지 다변량 활용 방식을 49-window 롤링 평가** 로 검증했다.
>
> **TimesFM 2.5 Zero-Shot 이 5가지 다변량 시도를 모두 능가** (+6.9~170% 차이).
>
> 이는 negative result 가 아니라 **foundation model 의 일반화가 본 농산물 데이터에서 명시적 covariate 결합보다 강력**하다는 정량적 증거이며, **운영 비용 0, retrain 불필요, 신규 품목 즉시 대응 가능한 ZS 를 최종 선정한 정당한 근거**다.
>
> **데이터셋 자체가 부실한 것이 아니다**: TTM FT 의 +57.4% 개선, Chronos2 LoRA 의 일관된 5~6% 개선, xreg 두 모드 비대칭 결과 — 이 세 증거가 데이터에 풍부한 covariate-target 신호가 있음을 정량적으로 증명한다. ZS 가 우세한 이유는 **사전학습이 그 신호를 이미 효과적으로 흡수**했기 때문이다.

---

## 7. 한계와 향후 과제

### 7-1. 점예측 (TimesFM ZS)

- **더 긴 context 미시도** — 모델은 16,384 까지 지원. 1024~1536 시도 가치 있음.
- **품목별 라우팅 미적용** — per_crop_winner 분석 결과 ZS 22 / LoRA 21 / xreg 3. 룰 라우팅 시 추가 이득 가능.
- **TimesFM 3.0 출시 대비** — base 모델 업그레이드 시 ZS 그대로 갱신 가능.

### 7-2. 구간예측 (Chronos2 LoRA)

- **Conformal recalibration (CQR) 미적용** — PICP 가 이미 목표 근접하므로 미적용. 더 엄격한 calibration 필요 시 적용 가능.
- **LoRA + ZS 앙상블 미실시** — 사양 제약. 운영 시 룰 라우팅 가능.

### 7-3. 일기예보 통합 (05_WEATHER_FORECAST)

- **`weather_sunshine_dur` 미지원** — 단기예보엔 SKY/PTY 만, 중기엔 없음. 현재 ffill. SKY/PTY → 일조시간 회귀 모델 필요.
- **`market_rest` 미통합** — 한국천문연구원 특일정보 API (`SpcdeInfoService`) 통합 필요.
- **산지 좌표 정밀도** — 시청 1개 행만 사용. 천안/전주/창원 등은 보정 가능성.
- **18시 발표 시 D+4 누락** — 06시 발표 우선 사용 권장.
- **xreg 변수 확장 미검증** — SKY, SNO, VEC 등 추가 변수 도입은 backtest 필요.
- **일기예보 vs 실측 백테스트 미수행** — 1주일 후 실제 관측값과의 정량 비교 권장.

### 7-4. 데이터 측면

- **신규 품목 학습** — TimesFM ZS 는 즉시 동작, Chronos2 LoRA 는 정확도 유지 위해 retrain 권장 (8~10분).
- **외부 도메인 확장** — 수산물/축산물 등 적용 시 LoRA adapter 와 covariate 정의 재설계 필요.
- **데이터 갱신 SOP** — 매일 도매가/거래량 append, 매월 1일 금리·CPI 갱신.

---

## 8. 부록 — 원본 데이터 소재 매핑

### 8-1. 모든 표·그래프의 원본 위치

| 항목 | 원본 파일 |
|---|---|
| 점예측 4-way 메트릭 | `03_POINT_FORECAST/experiments/metrics_table_4way.csv` |
| 점예측 7-way 메트릭 | `03_POINT_FORECAST/experiments/metrics_table_7way.csv` |
| 점예측 작물별 MASE | `03_POINT_FORECAST/experiments/per_crop_mase_*.csv` |
| 점예측 그리드 | `03_POINT_FORECAST/experiments/grid_timesfm.csv`, `grid_ttm.csv` |
| 점예측 xreg 윈도우 | `03_POINT_FORECAST/experiments/per_window_xreg_{a,b,static}.csv` |
| 점예측 49-win 백테스트 (NEW) | `tmp/tmp_timesf2.5/outputs/metrics_overall.csv`, `metrics_per_horizon.csv`, `metrics_per_item.csv`, `metrics_per_crop.csv` |
| 점예측 TTM 4-cov fair (NEW) | `tmp/tmp_ttm/outputs/metrics_overall.csv`, `comparison_with_timesfm_zs.csv` |
| 구간예측 메트릭 | `02_PROBABILISTIC_FORECAST/experiments/metrics_table.csv` |
| 구간예측 작물별 WQL | `02_PROBABILISTIC_FORECAST/experiments/per_crop_wql.csv` |
| 구간예측 품목 winrate | `02_PROBABILISTIC_FORECAST/experiments/per_item_winrate.csv` |
| 구간예측 LoRA vs ZS | `02_PROBABILISTIC_FORECAST/experiments/lora_vs_zs_per_crop.csv` |
| 일기예보 파일럿 결과 | `tmp/weather_xreg_pilot/outputs/chronos2_forecast_with_weather.parquet`, `timesfm_forecast_xreg.parquet`, `timesfm_forecast_zeroshot.parquet` |

### 8-2. 차트 위치

| 차트 | 경로 |
|---|---|
| 점예측 4-way 막대 | `03_POINT_FORECAST/experiments/figures/model_comparison_bar.png` |
| 점예측 7-way 막대 | `03_POINT_FORECAST/experiments/figures/model_comparison_7way_bar.png` |
| 점예측 작물 우승 (4-way) | `03_POINT_FORECAST/experiments/figures/per_crop_winner.png` |
| 점예측 작물 우승 (7-way) | `03_POINT_FORECAST/experiments/figures/per_crop_winner_7way.png` |
| 점예측 3-모델 비교 (8 품목 + grid) | `03_POINT_FORECAST/sample_forecasts/` |
| 구간예측 calibration | `02_PROBABILISTIC_FORECAST/experiments/figures/calibration_diagram.png` |
| 구간예측 variant radar | `02_PROBABILISTIC_FORECAST/experiments/figures/variant_radar.png` |
| 구간예측 작물 WQL | `02_PROBABILISTIC_FORECAST/experiments/figures/per_crop_wql.png` |
| 구간예측 9분위 pinball | `02_PROBABILISTIC_FORECAST/experiments/figures/per_quantile_pinball.png` |
| 구간예측 윈도우 추세 | `02_PROBABILISTIC_FORECAST/experiments/figures/per_window_trend.png` |
| 구간예측 PICP CQR 효과 | `02_PROBABILISTIC_FORECAST/experiments/figures/picp_cqr_effect.png` |
| 구간예측 13작물 fan chart | `02_PROBABILISTIC_FORECAST/sample_forecasts/` |
| 일기예보 fan chart 샘플 | `tmp/weather_xreg_pilot/samples/apple_fuji_box10kg_high_chronos2_fan.png` |
| TimesFM xreg vs ZS 샘플 | `tmp/weather_xreg_pilot/samples/apple_fuji_box10kg_high_timesfm_compare.png` |
| 13작물 grid (Chronos2 + 일기예보) | `tmp/weather_xreg_pilot/samples/grid_13crops_chronos2.png` |

### 8-3. 보고서 위치

| 문서 | 위치 |
|---|---|
| 통합 보고서 (narrative) | `00_INTEGRATED_REPORT.md` |
| 본 종합 보고서 (정량) | `EXPERIMENT_REPORT.md` (이 문서) |
| 6-way 비교 narrative | `02_PROBABILISTIC_FORECAST/experiments/6way_comparison.md` |
| 4-way 비교 narrative | `03_POINT_FORECAST/experiments/4way_comparison.md` |
| 7-way ablation narrative | `03_POINT_FORECAST/experiments/7way_ablation.md` |
| 그리드 서치 narrative | `03_POINT_FORECAST/experiments/grid_search.md` |
| 49-win 백테스트 narrative | `tmp/tmp_timesf2.5/REPORT.md` |
| TTM 4-cov fair narrative | `tmp/tmp_ttm/REPORT.md` |
| 일기예보 파일럿 narrative | `tmp/weather_xreg_pilot/REPORT.md` |
| 백엔드 인계 매뉴얼 | `BACKEND_HANDOVER.md` |
| 검증 실행 로그 | `VERIFICATION.md` |

---

## 발표 핵심 메시지 (3장)

### 슬라이드 1: "단변량을 안 쓴 것이 아니라, 다변량을 다 시도해본 결과 ZS가 우세"

- 다변량 5가지 시도 (xreg 3 + TTM 2) 모두 ZS 보다 악화 (+6.9~170%)
- 데이터 quality 의 정량적 반증 3가지 (TTM FT +57.4% / Chronos2 LoRA 일관 5~6% / xreg 비대칭)
- 사전학습이 covariate 신호를 이미 흡수했기 때문

### 슬라이드 2: "구간예측의 완벽한 calibration + 일기예보 통합"

- Chronos2 LoRA WQL 0.1298, PICP@60/80 = 0.589/0.789 (목표와 1.1pp 차이)
- 기상청 단기/중기 API 통합으로 운영 가능 (05_WEATHER_FORECAST 모듈)
- 일기예보를 점예측에는 미적용 (백테스트로 ZS 우세 확인), 구간예측에만 적용

### 슬라이드 3: "두 모델 병행 운영 — 학습 비용 / 적용 영역 명확"

- TimesFM 2.5 ZS: MASE 0.806, 학습 비용 0, 신규 품목 즉시 대응
- Chronos2 LoRA: WQL 0.1298, 학습 10분/variant, 분기별 retrain
- 점예측 ↔ 구간예측 용도 분리 (단일값 vs 신뢰구간)
