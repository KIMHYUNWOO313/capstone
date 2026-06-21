농산물 예측용 데이터·전처리·모델 코드를 이 폴더(또는 하위 디렉터리)에 두면 됩니다.
PUBG/ 와 동일한 패턴 예: AGRICULTURE/data/, AGRICULTURE/preprocessing/

원천 xlsx: **data/농넷 데이터/** (품목별 가격 시트만 사용).

0a) 농넷 원본 그대로 · 품종×거래단위×등급 분리 (결측·보간 없음, DATE만 오름차순)
   python AGRICULTURE/preprocessing/split_nongnet_raw_by_series.py
   - 예: 당근.xlsx → 당근_당근(전체)_10kg_중품.csv 등 (엑셀에 있는 조합만큼)
   - 산출: AGRICULTURE/data/processed/nongnet_by_series/
            split_nongnet_raw_manifest.json

0a-2) 평균가격만: 연속 결측이 **양쪽에 실제 값**이 있을 때만 채움
   python AGRICULTURE/preprocessing/fill_nongnet_price_interior_mean.py
   - 채움값 = int(round((바로 위 평균가격 + 바로 아래 평균가격) / 2)), 연속 구간 전체 동일
   - 입력: nongnet_by_series/*.csv → 출력: nongnet_by_series_filled/
            fill_nongnet_price_manifest.json

0) 농넷 xlsx 시계열용 결측 처리
   python AGRICULTURE/preprocessing/clean_woncheon_timeseries.py
   - 입력은 위 폴더를 우선 사용 (없으면 data/ 내 다른 .xlsx 폴더).
   - 산출: AGRICULTURE/data/processed/woncheon_cleaned/*.csv
            AGRICULTURE/data/processed/preprocessing_woncheon_manifest.json

0b) 거래단위·등급별 시계열 분리
   python AGRICULTURE/preprocessing/split_woncheon_by_unit_grade.py
   - 입력: woncheon_cleaned/*.csv
   - 산출: AGRICULTURE/data/processed/woncheon_by_series/
            split_woncheon_series_manifest.json

3) 지역별 농업기상 기본 관측 (공공데이터 OpenAPI)
   - 데이터셋: https://www.data.go.kr/data/15078057/openapi.do
   - 포털에서 활용신청 → DATA_GO_KR_SERVICE_KEY 설정
   - python AGRICULTURE/collect/fetch_agriweather_openapi.py --year-from 2023 --year-to 2023
   - 산출: AGRICULTURE/data/raw/agriweather/

(선택) 공공데이터 **연도별 도소매 CSV**를 data/ 루트에 다시 두면:
1) python AGRICULTURE/preprocessing/aggregate_major_crops_daily.py
2) python AGRICULTURE/visualization/plot_major_crop_trends.py

Chronos-2 연동 요약
- major_crops_chronos_long.csv: item_id, timestamp, target (aggregate 스크립트로 생성 시).
- 품목별 시계열은 woncheon_by_series CSV를 Chronos용 long 포맷으로 옮기면 됨.
