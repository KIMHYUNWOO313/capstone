# 05_WEATHER_FORECAST — 기상청 일기예보 → known_covariates 빌드 모듈

`02_PROBABILISTIC_FORECAST` (Chronos2 LoRA, 10일 확률예측) 의 `known_covariates`
중 미래값이 실제로 변동되는 `weather_temp_range` 를 **기상청 Open API** 로 채워
운영 환경에서 모델이 정상 동작하도록 한다.

> **점예측 (TimesFM 2.5) 에는 일기예보를 사용하지 않는다.** 49-window 백테스트
> 결과 xreg(oracle/noisy) 모드 모두 ZS 대비 MASE +4.8~4.9% 악화로, 미적용이 최적
> 이었기 때문 (`EXPERIMENT_REPORT.md` §3 참조).

---

## 1. 모듈 구성

| 파일 | 역할 |
|---|---|
| `stn_grid_map.py` | 산지 18개 영문명 → 단기예보 격자 `(nx, ny)` + 중기예보 시·군 구역 `regId` 매핑 |
| `weather_fetcher.py` | `getVilageFcst` / `getMidTa` 호출 + 1시간 단위 → 일별 집계 |
| `covariate_builder.py` | 46품목 × 10일 `known_covariates` DataFrame 빌드 (운영 진입점) |
| `.env` | 기상청 API 키 두 개 (`short_forecast_api`, `middle_forecast_api`) |
| `.env.example` | 키 발급 가이드 포함 템플릿 |

---

## 2. 빠른 사용 예 (백엔드 통합 코드)

```python
from pathlib import Path
import json, sys
import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

# 1) 05_WEATHER_FORECAST 모듈 import
WF_DIR = Path(__file__).parent.parent / "05_WEATHER_FORECAST"
sys.path.insert(0, str(WF_DIR))
from covariate_builder import build_known_covariates_frame, KNOWN_COVS

# 2) 모델 + 데이터 로드
predictor = TimeSeriesPredictor.load("../02_PROBABILISTIC_FORECAST/model")
with open("../01_DATA/meta_baseline.json", encoding="utf-8") as f:
    meta = json.load(f)
item_station_map = meta["item_station_map"]
full = pd.read_parquet("../01_DATA/data/full_baseline_extended_20260516.parquet")

# 3) 일기예보 기반 known_covariates 빌드
known_df = build_known_covariates_frame(full, item_station_map, horizon=10)
known_df = known_df[["item_id", "timestamp"] + KNOWN_COVS]
known_fc = TimeSeriesDataFrame.from_data_frame(known_df,
                id_column="item_id", timestamp_column="timestamp")

# 4) Chronos2 LoRA 추론
ts_full = TimeSeriesDataFrame(full.set_index(["item_id", "timestamp"])
                              if "item_id" in full.columns else full)
forecast = predictor.predict(ts_full, known_covariates=known_fc,
                              model="Chronos2LoRA_baseline")
print(forecast.head())
```

---

## 3. 기상청 두 API 사용 방식

### 3-1. 단기예보 (`getVilageFcst`, D+0~D+3)

- **호출 인자**: `nx`/`ny` (격자) + `base_date`/`base_time` (02·05·08·11·14·17·20·23시)
- **응답**: 14 카테고리 × 시간별 (한 번에 700~1000행, 페이지네이션 자동)
- **추출**: 일별 `TMN`/`TMX` (1건/일), `TMP`/`POP`/`REH`/`WSD` (평균)
- **활용 변수**: `temp_range = TMX - TMN` (Chronos2 known)

### 3-2. 중기기온 (`getMidTa`, D+4~D+10)

- **호출 인자**: `regId` (시·군 구역 8자) + `tmFc` (06·18시 발표)
- **응답**: `taMin{4..10}`, `taMax{4..10}` 한 건
- **활용 변수**: `temp_range = taMax - taMin` (Chronos2 known)

### 3-3. 발표시각 자동 선택

`weather_fetcher.py` 의 `latest_short_base()` / `latest_mid_base()` 가 현재 시각
기준 가장 최근 발표시각을 자동 계산. **18시 발표 사용 시 D+4 누락** → 코드는
"중기 → 단기 D+4 보조 → fallback" 순으로 우아하게 처리. 운영 시 가급적 **06시
발표** 이후(06:10~) 호출 권장.

---

## 4. `item_station_map` 매핑

`01_DATA/meta_baseline.json` 의 `item_station_map` 이 각 품목을 18개 산지 중
하나로 매핑한다 (train 기간 Spearman 상관 기반 선정). `stn_grid_map.py` 의
`STATION_TO_GRID` / `STATION_TO_MID_REGID` 가 산지명 → API 좌표를 변환.

### 18개 산지

| 산지(영문) | 한글 | nx, ny | regId |
|---|---|---|---|
| jecheon | 제천 | 81, 118 | 11C10201 |
| yeongju | 영주 | 89, 111 | 11H10401 |
| andong | 안동 | 91, 106 | 11H10501 |
| icheon | 이천 | 68, 121 | 11B20701 |
| cheonan | 천안 | 63, 110 | 11C20301 |
| geumsan | 금산 | 69, 95 | 11C20601 |
| jeonju | 전주 | 63, 89 | 11F10201 |
| haenam | 해남 | 54, 61 | 11F20302 |
| mokpo | 목포 | 50, 67 | 21F20801 |
| sangju | 상주 | 81, 102 | 11H10302 |
| pohang | 포항 | 102, 94 | 11H10201 |
| changwon | 창원 | 90, 77 | 11H20301 |
| sancheong | 산청 | 76, 80 | 11H20703 |
| miryang | 밀양 | 92, 83 | 11H20601 |
| jeju | 제주 | 53, 38 | 11G00201 |
| seongsan | 성산 | 60, 37 | 11G00101 |
| gangneung | 강릉 | 92, 131 | 11D20501 |
| daegwallyeong | 대관령 | 89, 130 | 11D20201 |

---

## 5. 운영 시 주의사항

1. **`full_baseline` 마지막 timestamp = today − 1** 이어야 horizon 일자와 일기
   예보 D+1~D+10 매핑이 정확. 매일 데이터 갱신 파이프라인 필수.
2. **단기예보 호출은 발표시각 +15분 이후**. `latest_short_base()` 가 자동 처리.
3. **중기예보는 06시 발표를 우선** 사용 (18시는 D+4 누락).
4. **호출 캐시**: 같은 `(nx, ny)` / `regId` 는 모듈 내 in-memory 캐시 사용.
   일일 일괄 갱신 시 `covariate_builder.clear_caches()` 로 초기화.
5. **API rate limit**: 30 tps. 46품목을 18개 산지로 그룹화하여 호출하므로 한
   번의 known 빌드에서 최대 36회 (단기 18 + 중기 18) 호출. 여유 충분.
6. **API 키 보안**: `.env` 는 절대 git 커밋 금지. 운영 환경은 OS 환경변수 또는
   비밀 관리 시스템(AWS Secrets Manager / GCP Secret Manager) 으로 주입.

---

## 6. 단위 검증

```bash
# 환경: anaconda capstone (PYTHONNOUSERSITE=1 권장)
cd c:/Users/minsoo/Desktop/chronos2/final_handoff/05_WEATHER_FORECAST

# 6-1. 매핑 일관성
python stn_grid_map.py
# → "산지 18개, 매핑 일관성 OK" + 18행 출력

# 6-2. API 호출 (단기 + 중기)
python weather_fetcher.py
# → 제천 단기예보 일별 5행 + 중기기온 D+4~D+10 일교차 7개

# 6-3. known_covariates 프레임
python covariate_builder.py
# → 46품목 × 10일 = 460행, NaN 0개, apple_fuji_box10kg_high 샘플 출력
```

---

## 7. 알려진 한계 (다음 작업 인계)

1. **`weather_sunshine_dur` 미지원**: 단기예보엔 `SKY`/`PTY` 만, 중기엔 일조시간
   변수가 없음. 현재는 마지막 관측치 ffill. SKY/PTY → 일조시간 회귀 모델 필요.
2. **`market_rest` 휴장일 미통합**: 한국천문연구원 특일정보 API
   (`SpcdeInfoService`) 통합 필요. 현재는 마지막 관측치(보통 0) ffill.
3. **18시 발표 시 D+4 누락**: 코드는 단기 D+4 보조 → fallback 으로 처리하나,
   운영 시 06시 발표를 우선 호출하도록 스케줄링 권장.
4. **산지 좌표 정밀도**: 시청 소재지 1개 행을 사용 (천안 동남구/서북구 등 분리
   되는 시·군은 첫 번째 행만). 산지 실제 위치와 미세 차이가 있으면 보정 필요.
5. **xreg 변수 확장**: SKY(하늘상태), SNO(신적설) 등 추가 변수 도입은 백테스트
   필요 (현재 4종으로도 ZS 미달이므로 우선순위 낮음).

---

## 8. 관련 문서

- `EXPERIMENT_REPORT.md` §5: 일기예보 통합 실험 (Chronos2 통합 검증, TimesFM
  xreg 백테스트 미적용 결정 근거)
- `BACKEND_HANDOVER.md` §3: API 키 발급·관리, §4: 일일 운영 워크플로우
- `99_REFERENCES/`: 기상청 단기/중기 API 활용가이드 docx + 격자/구역코드 엑셀
