# 농산물 가격 예측 — 종합 프로젝트 결과물

작성일: 2026-05-18 (일기예보 통합 + 종합 보고서 + 백엔드 인계 매뉴얼 반영)
대상: 발표·종합 산출물 (백엔드 운영 인수인계 포함)

본 폴더는 농산물 가격 예측 프로젝트의 **데이터 선정 → 전처리 → 모델 비교 → 파인튜닝 → 다변량 ablation → 일기예보 통합 → 최종 선정** 의 전 과정과 그 산출물을 하나로 묶은 종합 결과물입니다.

---

## 0. 폴더 한눈에 보기

```
final_handoff/
├── 00_README.md                ← 본 파일 (전체 인덱스)
├── 00_INTEGRATED_REPORT.md     ★ 종합 보고서 (전 과정 narrative)
├── EXPERIMENT_REPORT.md        ★ 발표용 종합 실험 결과 (정량 표·그래프 중심)
├── BACKEND_HANDOVER.md         ★ 백엔드 인계 매뉴얼 (운영 가이드)
├── VERIFICATION.md             ★ 실행 검증 로그
│
├── 01_DATA/                    ← 데이터 출처·전처리 6단계·variant 비교
├── 02_PROBABILISTIC_FORECAST/  ← 모델 1: Chronos2 LoRA (10일 확률범위) + 6-way 실험
├── 03_POINT_FORECAST/          ← 모델 2: TimesFM 2.5 ZS (3일 점예측) + 7-way ablation
├── 04_XAI_EXPLAINER/           ← 확장: GPT-4o 기반 사후 설명 모듈
├── 05_WEATHER_FORECAST/        ★ 신규: 기상청 일기예보 → known_covariates 빌더 (운영 필수)
└── 99_REFERENCES/              ← 외부 참고 자료 (기상청 API 가이드 등)
```

번호 prefix 는 **발표 순서**를 그대로 따릅니다 (데이터 → 모델 1 → 모델 2 → 확장 → 운영 모듈).

---

## 1. 한 페이지 요약

| 영역 | 결정 | 핵심 근거 |
|---|---|---|
| **데이터** | baseline variant (172,868 × 13, Known 5 + Past 7) | 3 variant 비교 — 도메인 지식 기반 baseline 이 자동 선정 optimal 보다 우수 ([`01_DATA`](01_DATA/)) |
| **모델 1 — 10일 확률범위** | **Chronos2 LoRA (baseline) + 기상청 일기예보** | 6 케이스 비교 1위, WQL 0.1298, PICP@60/@80 거의 완벽 calibration ([`02_PROBABILISTIC_FORECAST/experiments/6way_comparison.md`](02_PROBABILISTIC_FORECAST/experiments/6way_comparison.md)). 운영용 known_covariates 는 [`05_WEATHER_FORECAST`](05_WEATHER_FORECAST/) 모듈이 기상청 API 로 채움 |
| **모델 2 — 3일 점예측** | **TimesFM 2.5 Zero-Shot** (univariate) | 7 케이스 비교 1위, MASE 0.806. 49-window 백테스트로 일기예보 xreg 미적용 확정 (oracle/noisy 모두 +4.5/4.8% 악화). TTM 4-cov fair 비교도 +6.3% 악화 ([`EXPERIMENT_REPORT.md`](EXPERIMENT_REPORT.md) §3) |
| **운영 — 일기예보 통합** | 기상청 단기/중기 API → Chronos2 known | 18개 산지 매핑, 단기 D+1~D+3 + 중기 D+4~D+10 일교차, 일일 ~36 API 호출 ([`05_WEATHER_FORECAST/README.md`](05_WEATHER_FORECAST/README.md)) |
| **확장 — XAI 설명** | GPT-4o + Responses API + web_search | 사전 캐싱된 예측 parquet 기반, 농업월보·기상특보·웹검색을 종합한 한국어 사후 설명 ([`04_XAI_EXPLAINER`](04_XAI_EXPLAINER/)) |

---

## 2. 발표용 흐름 (권장 슬라이드 매핑)

| # | 슬라이드 주제 | 본 폴더 자료 |
|---|---|---|
| 1 | 문제 정의 — 농산물 가격 예측, 두 가지 horizon | [`00_INTEGRATED_REPORT.md`](00_INTEGRATED_REPORT.md) §1 |
| 2 | 데이터 출처 (5 도메인) + item_id 46개 설계 | [`01_DATA/README.md`](01_DATA/README.md) §1 |
| 3 | 전처리 6단계 + Known/Past 변수 분류 | [`01_DATA/README.md`](01_DATA/README.md) §2~3 |
| 4 | 데이터셋 variant 3종 비교 → baseline 채택 | [`01_DATA/README.md`](01_DATA/README.md) §4 |
| 5 | 모델 1 — Chronos2 LoRA 학습 셋업 | [`02_PROBABILISTIC_FORECAST/experiments/6way_comparison.md`](02_PROBABILISTIC_FORECAST/experiments/6way_comparison.md) §1 |
| 6 | 모델 1 — 6 케이스 비교 결과·핵심 발견 | 위 §2~3 |
| 7 | 모델 1 — Calibration 분석 (PICP) | 위 §6 + [`figures/calibration_diagram.png`](02_PROBABILISTIC_FORECAST/experiments/figures/calibration_diagram.png) |
| 8 | 모델 2 — TimesFM/TTM × ZS/FT 4-way 비교 | [`03_POINT_FORECAST/experiments/4way_comparison.md`](03_POINT_FORECAST/experiments/4way_comparison.md) |
| 9 | 모델 2 — 하이퍼파라미터 그리드 탐색 | [`03_POINT_FORECAST/experiments/grid_search.md`](03_POINT_FORECAST/experiments/grid_search.md) |
| 10 | 모델 2 — ★ 다변량 활용 5가지 ablation | [`03_POINT_FORECAST/experiments/7way_ablation.md`](03_POINT_FORECAST/experiments/7way_ablation.md) §1~3 |
| 11 | 모델 2 — 다변량이 ZS 못 넘은 인과 분석 | 위 §4 |
| 12 | 모델 2 — "데이터셋이 부실해서가 아니다" (방어 근거) | 위 §5~6 ★ |
| 13 | 최종 선정·운영 권장 | [`00_INTEGRATED_REPORT.md`](00_INTEGRATED_REPORT.md) §5 |
| 14 | 확장 — XAI 사후 설명 데모 | [`04_XAI_EXPLAINER/README.md`](04_XAI_EXPLAINER/README.md) §3 |
| 15 | 한계 + 후속 개선 아이디어 | [`00_INTEGRATED_REPORT.md`](00_INTEGRATED_REPORT.md) §7 |

---

## 3. 폴더별 산출물 요약

### [`01_DATA/`](01_DATA/)
- `README.md` — 원천 5 도메인 → 6단계 전처리 → 3 variant 비교 narrative
- `data/full_baseline.parquet` (1.9 MB, 172,868 × 13) — 메인 학습 데이터
- `data/static_baseline.parquet` (4.8 KB, 46 × 4) — 정적 속성
- `data/full_baseline_extended_20260516.parquet` (2.1 MB) — 갱신본 (참고)
- `meta_baseline.json` — known/past covariate 분류 명세
- `dataset_description_probabilistic.md` / `_point.md` — 변수 사전

### [`02_PROBABILISTIC_FORECAST/`](02_PROBABILISTIC_FORECAST/) — Chronos2 LoRA (10일)
- `README.md` — 모델 개요·셋업·선정 결과·운영 가이드
- `experiments/6way_comparison.md` ★ — variant 3 × {ZS, LoRA} 비교 narrative
- `experiments/*.csv` — 메트릭·작물별·LoRA vs ZS (4 CSV)
- `experiments/figures/*.png` — 7장 (calibration, radar, winrate, per_crop, per_window, pinball, cqr)
- `sample_forecasts/*.png` — 13작물 예측 시각화
- `predict_example.py` + `requirements.txt`
- `model/` — AutoGluon TimeSeriesPredictor 즉시 로드 (LoRA adapter 포함)

### [`03_POINT_FORECAST/`](03_POINT_FORECAST/) — TimesFM 2.5 ZS (3일)
- `README.md` — 모델 개요·"왜 단변량 ZS 인가"·운영 가이드
- `experiments/4way_comparison.md` — TimesFM/TTM × {ZS, FT} 4 케이스
- `experiments/7way_ablation.md` ★ — 다변량 활용 5가지 + ZS 정당화 + 데이터 quality 정량 반증
- `experiments/grid_search.md` — LoRA · TTM 하이퍼파라미터 탐색
- `experiments/*.csv` — 메트릭·MASE·grid·xreg 윈도우/summary (13 CSV)
- `experiments/figures/*.png` — 4장 (4-way·7-way 막대 + 4-way·7-way per_crop_winner)
- `sample_forecasts/*.png` — 3-모델 비교 (8 품목 + grid)
- `predict_example.py` + `requirements.txt`

### [`04_XAI_EXPLAINER/`](04_XAI_EXPLAINER/) — GPT-4o XAI 모듈
- `README.md` — 설치·실행·출력 스키마·운영 가이드
- `explain.py`, `run_xai.py`, `prompt_builder.py`, `config.py`, `stn_code_map.py` — 코어
- `data_loaders/` — forecasts·context·weather_warn·reports
- `inputs/` — 사전 캐싱된 forecast parquet + 농넷 월보 PDF 3개 (2026-05) + 변수 사전 사본
- `scripts/refresh_forecasts.py` — 두 모델 재실행해 forecast 갱신
- `outputs/` — 실행 시 `explanations_YYYYMMDD.json` 생성

### [`05_WEATHER_FORECAST/`](05_WEATHER_FORECAST/) — 기상청 일기예보 운영 모듈 (NEW)
- `README.md` — 사용법, 산지 18개 매핑 표, 운영 주의사항
- `stn_grid_map.py` — 산지 영문명 → 단기 (nx,ny) + 중기 regId
- `weather_fetcher.py` — getVilageFcst (단기, D+0~D+3) + getMidTa (중기, D+4~D+10) 호출
- `covariate_builder.py` — 46품목 × 10일 known_covariates DataFrame 빌드 (운영 진입점)
- `.env` / `.env.example` — 기상청 API 키 (단기·중기)
- `requirements.txt` — requests, python-dotenv

### [`99_REFERENCES/`](99_REFERENCES/) — 외부 참고
- 기상청 기상특보 조회서비스 Open API 활용 가이드 (docx)
- 특보참고사항 (docx)

---

## 4. 빠른 재현

각 모델 독립 실행:

```bash
# 0. 일기예보 모듈 (먼저 .env에 기상청 키 채울 것)
cd 05_WEATHER_FORECAST
pip install -r requirements.txt
python weather_fetcher.py          # 단기/중기 API 호출 검증
python covariate_builder.py        # 46품목 × 10일 known 프레임 검증

# 1. 확률범위 (일기예보 통합 forecast 모드)
cd ../02_PROBABILISTIC_FORECAST
pip install -r requirements.txt
python predict_example.py           # 기본: forecast 모드 (기상청 API)
# python predict_example.py lookup  # 오프라인 평가 모드

# 2. 점예측 (univariate ZS, 변경 없음)
cd ../03_POINT_FORECAST
pip install -r requirements.txt
python predict_example.py

# 3. XAI 설명 (선택)
cd ../04_XAI_EXPLAINER
cp .env.example .env       # API 키 직접 입력
pip install -r requirements.txt
python scripts/refresh_forecasts.py  # 두 모델 forecast 캐시 갱신
python run_xai.py --limit 3 --dry-run
```

> `02_` `03_` `04_` 의 `predict_example.py` / `refresh_forecasts.py` 는 `../01_DATA/data/full_baseline_extended_20260516.parquet` 를 참조하므로 **상대 경로상 같은 부모 폴더에 있어야** 합니다. `02_` 와 `04_` 가 `05_WEATHER_FORECAST` 도 형제 폴더로 필요합니다. 두 모델은 별도 conda 환경 권장 (`autogluon.timeseries==1.5.0` vs `timesfm`). 자세한 설치·운영은 [`BACKEND_HANDOVER.md`](BACKEND_HANDOVER.md) 참조.

---

## 5. 본 README 가 답하지 않는 질문은 [`00_INTEGRATED_REPORT.md`](00_INTEGRATED_REPORT.md) 에 있습니다

INTEGRATED_REPORT 목차:
1. 프로젝트 전체 흐름
2. 원천 데이터 → 최종 데이터 변환 과정
3. 범위예측 (확률범위) 모델 비교·선정 과정
4. 점예측 모델 비교·선정 과정
4.5. ★ 다변량 활용 Ablation — 7-way 비교 + ZS 선정 정당화
5. 최종 권장 / 통합 결론
6. 환경 / 운영 / 트러블슈팅
7. 한계점·후속 개선 아이디어

---

## 6. 변경 이력

- **2026-05-18** — **일기예보 통합 완성판**. tmp/ 실험 결과(49-win 백테스트·TTM 4-cov fair·일기예보 파일럿) 를 본 폴더에 통합. 신규 폴더 `05_WEATHER_FORECAST/` 추가. `02_PROBABILISTIC_FORECAST/predict_example.py` 운영 모드 지원. `04_XAI_EXPLAINER/scripts/refresh_forecasts.py` 경로 버그 수정 + 일기예보 통합. 신규 문서 3개 작성 (`EXPERIMENT_REPORT.md`, `BACKEND_HANDOVER.md`, `VERIFICATION.md`).
- **2026-05-17** — 백엔드 인계 모드 → **종합 프로젝트 결과물 모드** 로 재구조화. 번호 prefix (01~04, 99) 도입, 각 모델 폴더에 `experiments/` 서브폴더 신설, 데이터 폴더 통합, `.env` 키 외부 백업 후 제거, 신규 분석 문서 5개 작성.
- **2026-05-14** — 백엔드 인계 초기 패키징.
