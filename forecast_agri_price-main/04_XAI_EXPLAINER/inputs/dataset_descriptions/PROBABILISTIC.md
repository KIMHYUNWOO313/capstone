# Chronos2 LoRA — Baseline 데이터셋 구성 설명

## 파일 목록

| 파일 | 크기 | 설명 |
|---|---|---|
| `data/full_baseline.parquet` | 1.9 MB | 메인 시계열 데이터 (46개 품목 × 일별) |
| `data/static_baseline.parquet` | 4.8 KB | 품목별 정적 속성 (작물명·등급·주산지 등) |
| `data/meta_baseline.json` | 2.9 KB | known/past covariate 분류 명세 |

---

## 1. `full_baseline.parquet`

### 기본 정보

| 항목 | 값 |
|---|---|
| 행 수 | 172,868 |
| 열 수 | 13 (target + 12 covariates) |
| 인덱스 | `(item_id, timestamp)` — MultiIndex |
| 기간 | 2015-11-16 ~ 2026-02-28 (일별, freq=D) |
| 품목 수 | 46개 (13개 작물 × 1~4개 등급) |

### 인덱스 컬럼

| 컬럼 | 타입 | 설명 |
|---|---|---|
| `item_id` | string | `{작물}_{규격}_{등급}` 형식 (예: `apple_fuji_box10kg_high`) |
| `timestamp` | datetime64 | 날짜 (일별, KST 기준) |

### 데이터 컬럼

#### 예측 대상

| 컬럼 | 타입 | 단위 | 설명 |
|---|---|---|---|
| `target` | float64 | 원(₩) | 농산물 도매 가격 (해당 규격·등급 1단위당) |

#### Known Covariates — 예측 구간(10일 후)에도 값을 알 수 있는 변수

| 컬럼 | 타입 | 설명 | 미래 값 획득 방법 |
|---|---|---|---|
| `market_rest` | int64 | 농수산물 도매시장 휴장일 여부 (1=휴장, 0=정상) | 공휴일 캘린더 |
| `weather_temp_range` | float64 | 주산지 일교차 (°C) | 기상청 7~10일 예보 |
| `weather_sunshine_dur` | float64 | 주산지 일조 시간 (hr) | 기상청 예보 |
| `bok_base_rate` | float64 | 한국은행 기준금리 (%) | 금통위 발표 (월 1회, ffill 유효) |
| `cpi_growth_rate` | float64 | 소비자물가 상승률 (%) | 통계청 월별 발표 (ffill 유효) |

#### Past Covariates — 예측 시점까지의 과거값만 사용 가능한 변수

| 컬럼 | 타입 | 설명 |
|---|---|---|
| `amount` | float64 | 거래량 (단위: 거래 건수 또는 kg, 품목마다 다름) |
| `oil_tax_free_diesel` | float64 | 면세 경유 가격 (원/L) — 매일 변동 |
| `weather_rain_sum` | float64 | 주산지 일강수량 (mm) |
| `weather_wind_avg` | float64 | 주산지 평균 풍속 (m/s) |
| `weather_humidity_avg` | float64 | 주산지 평균 습도 (%) |
| `weather_pressure_avg` | float64 | 주산지 평균 기압 (hPa) |
| `news_sentiment_index` | float64 | 농업 관련 뉴스 감성 지수 |

> **주의**: known/past 구분은 AutoGluon TimeSeriesPredictor 학습·예측 시 반드시 지켜야 함.
> known은 `known_covariates_names`에, past는 `past_covariates` 위치에 자동 배치됨 (target 이외 컬럼 전체).

---

## 2. `static_baseline.parquet`

품목당 1행 (46행 × 5열). 시간에 따라 변하지 않는 속성.

| 컬럼 | 타입 | 설명 | 값 예시 |
|---|---|---|---|
| `item_id` | string | 고유 품목 ID (full_baseline.parquet 의 item_id 와 동일) | `apple_fuji_box10kg_high` |
| `crop` | string | 작물명 (영문 소문자) | `apple`, `cabbage`, `carrot`, ... |
| `grade` | string | 등급 | `high`, `mid`, `low`, `premium` |
| `crop_group` | string | 작물 대분류 | `fruit`, `leaf`, `root`, `bulb` |
| `weather_station` | string | 해당 품목에 배정된 주산지 기상 관측소 | `jecheon`, `jeju`, `seongsan`, ... |

### item_id 명명 규칙

```
{작물명}_{규격}_{등급}
예: apple_fuji_box10kg_high
    ↑       ↑          ↑
    작물   규격·포장   등급
```

### 전체 46개 품목

| crop | 규격 | 등급 수 |
|---|---|---|
| apple | fuji_box10kg | 4 (high/mid/low/premium) |
| cabbage | net8kg | 4 |
| carrot | box20kg | 3 (high/mid/low) |
| crown_daisy | box4kg | 3 |
| cucumber_bdadagi | ea100 | 4 |
| garlic_chive | bundle500g | 3 |
| honewort | kg4 | 3 |
| napa_cabbage | net10kg | 4 |
| onion | kg1 | 4 |
| perilla_leaf | bunch100 | 3 |
| potato_sumi | box20kg | 4 |
| spinach | box4kg | 3 |
| sweetpotato | box10kg | 4 |

---

## 3. `meta_baseline.json`

```json
{
  "variant": "baseline",
  "known_covariates": ["market_rest", "weather_temp_range", "weather_sunshine_dur",
                       "bok_base_rate", "cpi_growth_rate"],
  "past_covariates":  ["amount", "oil_tax_free_diesel", "weather_rain_sum",
                       "weather_wind_avg", "weather_humidity_avg",
                       "weather_pressure_avg", "news_sentiment_index"],
  "item_station_map": { "apple_fuji_box10kg_high": "jecheon", ... },
  "train_end": "2024-02-29"
}
```

`item_station_map` 은 품목별 주산지 기상 관측소 매핑으로,
예측 시 기상청 예보를 품목별로 연결할 때 사용.

---

## 4. train / test 분할 기준

| 구간 | 기간 | 역할 |
|---|---|---|
| Train | 2015-11-16 ~ 2024-02-29 | 모델 학습 |
| Test | 2024-03-01 ~ 2026-02-28 | 롤링 윈도우 평가 (731일, 49 windows) |

데이터 파일에는 train+test 전체 기간이 포함되어 있음.
`TimeSeriesPredictor`의 `train_test_split(731)` 으로 분리 가능.

---

## 5. 모델 예측 출력 형식

`predictor.predict()` 결과 컬럼:

| 컬럼 | 설명 |
|---|---|
| `0.1` ~ `0.9` | 각 분위수 예측값 (9개) |
| `mean` | 평균 예측값 |

**80% 신뢰구간**: `0.1` (하한) ~ `0.9` (상한)  
**60% 신뢰구간**: `0.2` (하한) ~ `0.8` (상한)  
**중앙값 (점예측 대용)**: `0.5`

---

## 6. 품목별 주산지 매핑 (item_station_map)

각 품목에 배정된 기상 관측소는 train 구간에서 Spearman 상관분석으로 선정된 **최적 주산지**.
`weather_*` 컬럼 값은 품목별로 다른 관측소 데이터임.

| item_id | 작물 (crop) | 등급 (grade) | 주산지 관측소 |
|---|---|---|---|
| apple_fuji_box10kg_high | apple | high | **jecheon** (제천) |
| apple_fuji_box10kg_mid | apple | mid | **yeongju** (영주) |
| apple_fuji_box10kg_low | apple | low | **yeongju** (영주) |
| apple_fuji_box10kg_premium | apple | premium | **andong** (안동) |
| cabbage_net8kg_high | cabbage | high | **jeju** (제주) |
| cabbage_net8kg_mid | cabbage | mid | **jeju** (제주) |
| cabbage_net8kg_low | cabbage | low | **jeju** (제주) |
| cabbage_net8kg_premium | cabbage | premium | **jeju** (제주) |
| carrot_box20kg_high | carrot | high | **seongsan** (성산, 제주) |
| carrot_box20kg_mid | carrot | mid | **seongsan** (성산, 제주) |
| carrot_box20kg_low | carrot | low | **seongsan** (성산, 제주) |
| crown_daisy_box4kg_high | crown_daisy | high | **icheon** (이천) |
| crown_daisy_box4kg_mid | crown_daisy | mid | **icheon** (이천) |
| crown_daisy_box4kg_low | crown_daisy | low | **icheon** (이천) |
| cucumber_bdadagi_ea100_high | cucumber_bdadagi | high | **cheonan** (천안) |
| cucumber_bdadagi_ea100_mid | cucumber_bdadagi | mid | **cheonan** (천안) |
| cucumber_bdadagi_ea100_low | cucumber_bdadagi | low | **sangju** (상주) |
| cucumber_bdadagi_ea100_premium | cucumber_bdadagi | premium | **cheonan** (천안) |
| garlic_chive_bundle500g_high | garlic_chive | high | **changwon** (창원) |
| garlic_chive_bundle500g_mid | garlic_chive | mid | **changwon** (창원) |
| garlic_chive_bundle500g_low | garlic_chive | low | **changwon** (창원) |
| honewort_kg4_high | honewort | high | **sancheong** (산청) |
| honewort_kg4_mid | honewort | mid | **sancheong** (산청) |
| honewort_kg4_low | honewort | low | **sancheong** (산청) |
| napa_cabbage_net10kg_high | napa_cabbage | high | **gangneung** (강릉) |
| napa_cabbage_net10kg_mid | napa_cabbage | mid | **gangneung** (강릉) |
| napa_cabbage_net10kg_low | napa_cabbage | low | **daegwallyeong** (대관령) |
| napa_cabbage_net10kg_premium | napa_cabbage | premium | **gangneung** (강릉) |
| onion_kg1_high | onion | high | **changwon** (창원) |
| onion_kg1_mid | onion | mid | **changwon** (창원) |
| onion_kg1_low | onion | low | **changwon** (창원) |
| onion_kg1_premium | onion | premium | **changwon** (창원) |
| perilla_leaf_bunch100_high | perilla_leaf | high | **geumsan** (금산) |
| perilla_leaf_bunch100_mid | perilla_leaf | mid | **geumsan** (금산) |
| perilla_leaf_bunch100_low | perilla_leaf | low | **geumsan** (금산) |
| potato_sumi_box20kg_high | potato_sumi | high | **jeonju** (전주) |
| potato_sumi_box20kg_mid | potato_sumi | mid | **miryang** (밀양) |
| potato_sumi_box20kg_low | potato_sumi | low | **miryang** (밀양) |
| potato_sumi_box20kg_premium | potato_sumi | premium | **jeonju** (전주) |
| spinach_box4kg_high | spinach | high | **pohang** (포항) |
| spinach_box4kg_mid | spinach | mid | **pohang** (포항) |
| spinach_box4kg_low | spinach | low | **pohang** (포항) |
| sweetpotato_box10kg_high | sweetpotato | high | **haenam** (해남) |
| sweetpotato_box10kg_mid | sweetpotato | mid | **haenam** (해남) |
| sweetpotato_box10kg_low | sweetpotato | low | **haenam** (해남) |
| sweetpotato_box10kg_premium | sweetpotato | premium | **mokpo** (목포) |

> 같은 작물이라도 등급에 따라 다른 산지가 배정될 수 있음 (예: 배추 저등급은 대관령, 나머지는 강릉).

---

## 7. 의존 패키지 (AWS 환경 설치 필요)

```
autogluon.timeseries==1.5.0
torch>=2.0  (CUDA 12.x 권장, CPU만으로도 예측 가능)
pandas>=2.0
pyarrow>=14.0
numpy>=1.26
```

GPU 없이 CPU 전용 예측도 가능하나 배치 추론 속도는 GPU 대비 약 5~10배 느림.
