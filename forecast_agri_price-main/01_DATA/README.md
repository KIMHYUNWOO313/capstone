# 01_DATA — 데이터 선정·전처리 과정

본 폴더는 두 예측 모델(확률범위·점예측)이 **공통으로 사용하는 최종 학습 데이터**와 그 변환 과정을 정리합니다.

> 종합 보고서 [`../00_INTEGRATED_REPORT.md`](../00_INTEGRATED_REPORT.md) §2 의 폴더 단위 요약본입니다.

---

## 0. 핵심 요약

- **품목 수**: 46 (13개 작물 × 1~4개 등급)
- **기간**: 2015-11-16 ~ 2026-02-28 (일별, 10년+)
- **컬럼**: target 1 + covariates 12 (known 5, past 7) + 정적 속성 4
- **분할**: `train_end = 2024-02-29`, `test = 2024-03-01 ~ 2026-02-28` (731일)
- **롤링 평가**: cutoff step = 15일, 총 49 윈도우

---

## 1. 원천 데이터 — 5개 도메인

| 도메인 | 출처 | 원천 형식 |
|---|---|---|
| 농산물 가격 | aT 농수산식품유통공사 도매시장 통계 | 일별 도매 평균가, 거래량 |
| 기상 | 기상청 지상관측 (품목별 주산지) | 일평균 기온, 일교차, 강수량, 풍속, 습도, 기압, 일조시간 |
| 거시경제 | 한국은행 / 통계청 | 기준금리, CPI, M2, 국고채 3년 |
| 에너지 | 한국석유공사 | 면세 경유 가격 |
| 뉴스 감성 | 자체 크롤링 + 감성 분석 | 농업 뉴스 감성 지수 (일별) |

---

## 2. 원천 → 최종 데이터 변환 6단계

```
raw CSV (5 도메인)
  │
  │ ① item_id 생성 — 작물·규격·등급 단위 (46개)
  ▼
  품목별 일별 raw target
  │
  │ ② 주산지 매핑 — train 기간 Spearman 상관으로 품목별 최적 station 선정
  ▼
  품목 × 일별 (target + 12 covariates)
  │
  │ ③ 결측 처리 — target/amount: 선형보간 / 그 외: ffill
  │   ※ 사용자 검증: 보간이 NaN 보존보다 Chronos2 성능 우수
  │
  │ ④ Known / Past covariate 분류 정정
  │   - Known(5): market_rest, weather_temp_range, weather_sunshine_dur, bok_base_rate, cpi_growth_rate
  │   - Past(7) : amount, oil_tax_free_diesel, weather_{rain,wind,humidity,pressure}, news_sentiment
  │   ※ v9_1 의 oil_tax_free_diesel 잘못된 known 분류 정정 (매일 변동 → past)
  │
  │ ⑤ Train/Test 누설 검증
  │   - train_end(2024-02-29) ↔ test_start(2024-03-01) 사이 값 불연속 확인
  │   - v9_1 의 보간 누설 버그 차단
  │
  │ ⑥ Parquet 저장 (MultiIndex item_id × timestamp)
  ▼
  full_baseline.parquet (172,868 × 13) ← 최종 학습 데이터
  + static_baseline.parquet (46 × 정적속성)
  + meta_baseline.json (known/past 명세)
```

---

## 3. 변수 분류 결과

### Known Covariates (5개) — 예측 구간(미래 N일)에도 값 확보 가능

| 컬럼 | 단위 | 미래 값 획득 방법 |
|---|---|---|
| `market_rest` | 0/1 | 도매시장 공시 캘린더 |
| `weather_temp_range` | °C | 기상청 7~10일 예보 |
| `weather_sunshine_dur` | hr | 기상청 예보 |
| `bok_base_rate` | % | 금통위 발표 (월 1회, ffill) |
| `cpi_growth_rate` | % | 통계청 월별 발표 (ffill) |

### Past Covariates (7개) — 예측 시점까지의 과거값만 사용 가능

| 컬럼 | 단위 | 비고 |
|---|---|---|
| `amount` | 건/kg | 품목별 단위 상이 |
| `oil_tax_free_diesel` | 원/L | v9_1 에서 known→past 정정 |
| `weather_rain_sum` | mm | |
| `weather_wind_avg` | m/s | |
| `weather_humidity_avg` | % | |
| `weather_pressure_avg` | hPa | |
| `news_sentiment_index` | -1~1 | 자체 감성분석 |

### 정적 속성 (static_baseline.parquet — 4개)

| 컬럼 | 설명 |
|---|---|
| `crop` | 작물명 (13종) |
| `grade` | 등급 (high/mid/low) |
| `crop_group` | 작물군 (과실/엽채/근채 등) |
| `weather_station` | 매핑된 주산지 관측소 |

상세 변수 사전: [`dataset_description_probabilistic.md`](dataset_description_probabilistic.md) (확률예측 관점), [`dataset_description_point.md`](dataset_description_point.md) (점예측 관점).

---

## 4. 데이터셋 variant 비교 — 3종 vs 최종 선정

확률범위 모델 학습 시 다음 3개 데이터셋 variant 를 모두 학습·평가한 뒤 baseline 을 최종 채택:

| variant | rows | cols | 구성 의도 | Known | Past |
|---|---|---|---|---|---|
| **★ baseline** | 172,868 | 13 | 도메인 지식 기반 일반 구성 | 5 | 7 |
| no_weather | 172,868 | 7 | 기상 컬럼 전부 제거 (가설: 기상 영향 작음) | 3 | 3 |
| optimal | 172,868 | 15 | Spearman + SHAP + VIF 로 자동 선정 | 4 | 10 |

→ **최종 채택: baseline**. 근거는 [`../02_PROBABILISTIC_FORECAST/experiments/`](../02_PROBABILISTIC_FORECAST/experiments/) 의 6-way 비교 결과.

| 결과 요약 (LoRA 기준) | baseline | no_weather | optimal |
|---|---:|---:|---:|
| WQL ↓ | **0.1298** | 0.1301 (+0.2%) | 0.1326 (+2.2%) |
| PICP@80 (목표 0.8) | **0.789** | 0.772 | 0.774 |

핵심 관찰:
- 도메인 지식 기반 baseline 이 자동 선정 optimal 보다 우수 → **자동 feature 선정의 함정** 사례
- no_weather 가 baseline 과 사실상 동률 → 기상 변수의 기여도가 작음을 정량적으로 재확인 (단순화 관점에선 no_weather 도 후보)

---

## 5. 데이터 가공 요약 (최종본 적용)

| 항목 | 처리 방식 |
|---|---|
| 결측치 | target/amount: 선형보간 / 그 외: ffill |
| 이상치 | 자동 제거 없음 (모델이 처리) |
| 정규화 | raw 값 유지 (모델이 내부 RevIN/scaler 적용) |
| 시간 정렬 | 일별 (freq=D, KST) |
| 단위 통일 | 가격(원), 기상(°C·hr·mm·m/s·%·hPa), 금리(%), 감성지수(-1~1) |
| 주산지 매핑 | 품목별 Spearman 최적 station (item_station_map) |

---

## 6. 파일 안내

| 파일 | 크기 | 용도 |
|---|---:|---|
| `data/full_baseline.parquet` | 1.9 MB | 메인 학습 데이터 (172,868 × 13) |
| `data/full_baseline_extended_20260516.parquet` | 2.1 MB | 확장본 (참고용 — 일자 추가 갱신본) |
| `data/static_baseline.parquet` | 4.8 KB | 품목별 정적 속성 (46 × 4) |
| `meta_baseline.json` | 2.9 KB | known/past covariate 분류 명세 |
| `dataset_description_probabilistic.md` | — | 확률예측 관점 변수 사전 (가장 상세) |
| `dataset_description_point.md` | — | 점예측 관점 변수 사전 (univariate 사용 안내) |

---

## 7. 두 모델이 이 데이터를 어떻게 쓰는가

| 모델 | 사용 변수 | 비고 |
|---|---|---|
| [`02_PROBABILISTIC_FORECAST`](../02_PROBABILISTIC_FORECAST) (Chronos2 LoRA) | target + Known 5 + Past 7 | known covariate 운영 시 직접 채워 넣음 |
| [`03_POINT_FORECAST`](../03_POINT_FORECAST) (TimesFM 2.5 ZS) | target 만 (univariate) | covariate 컬럼 있어도 무시 — 단, 다변량 ablation §4.5 에서 5가지 결합 모두 시험함 |

> 점예측이 단변량인 것은 데이터의 부실 때문이 아니라 **5가지 다변량 활용 방식 ablation 결과 ZS 가 우세함을 정량 확인했기 때문**입니다. [`../03_POINT_FORECAST/experiments/7way_ablation.md`](../03_POINT_FORECAST/experiments/7way_ablation.md) 참조.
