# Final Handoff — 농산물 가격 예측 백엔드 인수인계

작성일: 2026-05-14
대상: 백엔드 팀

본 폴더는 두 종류의 예측 모델을 **즉시 운영 가능한 상태로 패키징**한 것입니다.

| 모듈 | 용도 | 모델 | 핵심 지표 |
|---|---|---|---|
| [`PROBABILISTIC_FORECAST/`](PROBABILISTIC_FORECAST/) | **10일 확률범위 예측** (9분위 + mean) | Chronos2 LoRA (baseline variant) | WQL 0.1298, PICP@80 0.789 |
| [`POINT_FORECAST/`](POINT_FORECAST/) | **3일 점예측** (mean) | TimesFM 2.5 Zero-Shot | MASE 0.806, MAPE 12.64% |

두 모델은 **같은 원천 데이터 (`full_baseline.parquet`)** 를 사용합니다. 데이터 파일은 양쪽 폴더에 중복 배치되어 폴더만 따로 옮겨도 동작합니다.

---

## 어떤 모델을 언제 쓰나

- **단일 점예측 한 숫자만 필요** (예: 일별 평균 가격 보드, MAPE/MASE 평가, 평균 손실 최소화) → **POINT_FORECAST**
- **신뢰구간이 필요** (예: 가격 변동 리스크 알람, 상하한 시나리오, 보험·헤지) → **PROBABILISTIC_FORECAST**
- 둘 다 운영해도 모순 없음. 점예측 모델이 더 정확하지만 신뢰구간을 제공하지 않으므로 용도 다름.

---

## 빠른 시작

```bash
# 1) 확률범위 예측
cd PROBABILISTIC_FORECAST
pip install -r requirements.txt
python predict_example.py

# 2) 점예측
cd ../POINT_FORECAST
pip install -r requirements.txt
python predict_example.py
```

> 두 모델은 별도 의존성 (`autogluon.timeseries==1.5.0` vs `timesfm`) 을 사용하므로 운영 시 **conda 환경 2개** 분리 권장. 한 환경에 둘 다 넣을 수도 있으나 transformers 버전 충돌 위험 (자세한 내용은 [`INTEGRATED_REPORT.md`](INTEGRATED_REPORT.md) §6 참조).

---

## 폴더 구조

```
final_handoff/
├── README.md                   # ← 본 파일 (인덱스)
├── INTEGRATED_REPORT.md        # ★ 전 과정 상세 보고 (필독)
│
├── PROBABILISTIC_FORECAST/     # 10일 확률범위 예측
│   ├── README.md
│   ├── dataset_description.md
│   ├── predict_example.py
│   ├── requirements.txt
│   ├── data/                   # 1.9MB
│   └── model/                  # AutoGluon predictor + LoRA adapter
│
├── POINT_FORECAST/             # 3일 점예측
│   ├── README.md
│   ├── dataset_description.md
│   ├── predict_example.py
│   ├── requirements.txt
│   └── data/                   # 1.9MB (PROBABILISTIC_FORECAST 와 같은 파일)
│
├── visualizations/             # 시각자료 모음
│   ├── prob/                   # 확률범위 모델 비교 차트 (7개)
│   ├── point/                  # 점예측 모델 비교 차트 (2개)
│   └── sample_forecasts/       # 품목별 예측 예시 차트 (13개)
│
└── evaluation_data/            # 평가 수치 원본 CSV (보고서 부록)
    ├── prob_metrics_table.csv          # 6 케이스 (3 variant × ZS/LoRA)
    ├── prob_per_crop_wql.csv           # 작물별 WQL
    ├── prob_per_item_winrate.csv       # 품목별 winrate
    ├── prob_lora_vs_zs_per_crop.csv    # LoRA vs ZS 개선율
    ├── point_metrics_table.csv         # 4 케이스 (TimesFM ZS/LoRA + TTM ZS/FT)
    ├── point_improvement_pct.csv       # FT 개선율
    ├── point_per_crop_mase.csv         # 품목별 MASE
    ├── point_ttm_grid.csv              # TTM 그리드 서치 로그
    └── point_timesfm_grid.csv          # TimesFM LoRA 그리드 서치 로그
```

---

## 본 README 가 답하지 않는 질문은 [`INTEGRATED_REPORT.md`](INTEGRATED_REPORT.md) 에 있습니다

INTEGRATED_REPORT.md 목차:
1. 프로젝트 전체 흐름
2. 원천 데이터 → 최종 데이터 변환 과정
3. 범위예측 (확률범위) 모델 비교·선정 과정
4. 점예측 모델 비교·선정 과정
5. 최종 권장 / 통합 결론
6. 환경 / 운영 / 트러블슈팅
7. 한계점·후속 개선 아이디어
