# 04_XAI_EXPLAINER — 농산물 가격 예측 설명 (XAI)

> **확장 산출물** — [`../02_PROBABILISTIC_FORECAST`](../02_PROBABILISTIC_FORECAST) (Chronos2 LoRA · 10일 확률예측) 과 [`../03_POINT_FORECAST`](../03_POINT_FORECAST) (TimesFM 2.5 · 3일 점예측) 의 결과를 받아 GPT-4o 가 "왜 이 가격이 예측되었는가" 를 한국어로 사후 설명하는 모듈입니다.

본 폴더만 받으면 즉시 동작하도록 설계되었습니다 (모델 파일·환경 불필요, 사전 캐싱된 예측 parquet 사용).

> ⚠ **API 키 안내**: 배포본에는 `.env` 파일이 포함되어 있지 않습니다 (보안). 사용 시 [`.env.example`](.env.example) 를 `.env` 로 복사 후 본인의 `GPT_API_KEY` 와 `WARN_API_KEY` 를 채우세요.

> **★ XAI의 목적**: 시계열 모델(Chronos2, TimesFM)의 가격 예측 결과를 **결론으로 받아들이고**,
> 그 결론이 왜 나왔는지를 모델 입력 데이터와 추가 자료(농업 월보·기상특보·web_search)를 근거로
> **사후 설명**합니다.
>
> 출력은 두 문단입니다:
> - **`forecast_summary`** — 모델 예측(3일/10일)을 1~2문장으로 요약
> - **`forecast_explanation`** — 위 결론이 왜 나왔는지를 2문단으로 설명
>
> 자세한 출력 스키마는 [§3. 출력 형식](#3-출력-형식-품목별-json) 참고.

---

## 1. 1회 설정

### (1) API 키 입력

`.env.example` 를 `.env` 로 복사한 뒤 두 키를 채우세요.

```
GPT_API_KEY=sk-...           # OpenAI API 키
WARN_API_KEY=...             # 공공데이터포털 기상특보 서비스키
```

선택 변수:
```
GPT_MODEL=gpt-4o             # 기본 gpt-4o, 비용 절감 시 gpt-4o-mini
```

> **보안 주의**: `.env` 파일에 본인의 OpenAI API 키가 들어 있다면, 패키지를 다른 사람에게 전달할 때 그 키도 함께 노출됩니다. 팀원과 키를 공유해도 무방하다면 `.env` 를 그대로 동봉하고, 각자 별도 키를 쓰는 게 안전하다면 `.env` 를 삭제한 뒤 `.env.example` 만 남겨 전달하세요.

### (2) 패키지 설치

Python 3.10+ 권장.

```bash
pip install -r requirements.txt
```

`autogluon` / `timesfm` 은 포함하지 않습니다 — 본 패키지는 사전 캐싱된 예측 결과만 읽으므로 모델 실행 환경이 필요 없습니다.

---

## 2. 실행

```bash
# 전체 46품목 일괄 설명
python run_xai.py

# 특정 품목만
python run_xai.py --items apple_fuji_box10kg_high,onion_kg1_high

# 테스트: 3개 품목, 프롬프트만 저장하고 GPT 호출 생략
python run_xai.py --limit 3 --dry-run

# 웹 검색 비활성화 (비용 절감, 보유 자료만 사용)
python run_xai.py --no-web-search

# 기상특보 API 호출 건너뛰기 (특보 API 장애 시)
python run_xai.py --skip-warnings
```

결과: `outputs/explanations_YYYYMMDD.json`

---

## 3. 출력 형식 (품목별 JSON)

품목별로 다음 스키마의 JSON 객체가 생성됩니다. **프론트엔드는 두 문단(`forecast_summary` + `forecast_explanation`)만 표시하면 됩니다.**

| 필드 | 타입 | 설명 |
|---|---|---|
| `item_id` | string | 품목 식별자 |
| `forecast_summary` ★ | string | **[1문단] 1~2문장.** 모델 예측(3일 점예측 + 10일 범위)을 한국어로 자연스럽게 요약. 이것이 explanation에서 설명해야 할 "결론"이 됨. |
| `forecast_explanation` ★ | string | **[2문단].** forecast_summary의 예측이 왜 나왔는지 사후 설명. 3일 점예측과 10일 범위 예측의 근거를 각각 짚어 산문으로 서술. 단락 사이는 `\n\n`. |
| `web_sources` | array | `web_search` 인용 출처. 빈 배열 `[]`이면 사용되지 않았거나 검증 미통과. 각 원소는 `{title, url, snippet, verified_date_range, topic_relevance}`. |
| `_meta` | object | `{model, web_search_calls, response_id}` — 디버깅용. |

### web_search 검증 규칙 (system prompt에 강제)

LLM이 `web_search` 도구를 사용할 때는 결과를 인용하기 전에 **다음 두 검증을 반드시 통과**해야 합니다:

1. **기간 검증** (`verified_date_range`): 검색 결과의 발행/취재 시점이 예측 기준 시점(현재)으로부터 **최근 30일 이내**인가? 작년·전 시즌·아카이브 자료는 인용 금지.
2. **주제 부합성 검증** (`topic_relevance`): 한국 농산물 도매가격·수급·작황·정책·기상 재해 주제에 **직접 관련**되는가? 가공식품, 해외 농산물, 무관 뉴스는 금지.

검증을 통과한 결과만 `web_sources`에 들어가고 검증 근거가 두 필드에 명시됩니다. 통과한 결과가 없으면 `web_sources`는 `[]`로 비어 있습니다.

### 실제 출력 예시 (`apple_fuji_box10kg_high`)

```json
{
  "item_id": "apple_fuji_box10kg_high",
  "forecast_summary": "사과 후지 10kg(상)은 향후 3일 동안 약 ₩62,600 수준의 가격이 예상되며, 향후 10일 동안에는 ₩36,270(하한)~₩75,541(상한) 범위 내에서 가격이 움직일 것으로 전망됩니다.",
  "forecast_explanation": "모델의 향후 3일 가격 예측이 약 ₩62,600으로 나타난 것은 최근 7일 동안의 평균 가격 ₩65,143이 다소 하락하는 추세를 보였으나 여전히 안정적이기 때문으로 추정됩니다. 또한, 기상특보가 최근 1주일 동안 발효되지 않은 점에서 특별한 가격 변동 압력이 적은 것으로 보입니다.\n\n향후 10일 가격 범위가 넓게 형성된 이유는 KREI 5월 농업관측 자료에 의하면 5월 사과 출하량이 전년 대비 4.6% 증가할 전망이므로 시장 공급이 충분하여 가격 변동성이 증가할 가능성이 있다고 추정됩니다. 특히, 최근 거래량이 365일 평균의 절반 수준으로 감소하면서, 공급의 변동성이 가격 변동에 큰 기여를 할 것으로 보입니다.",
  "web_sources": [],
  "_meta": {"model": "gpt-4o", "web_search_calls": 0, "response_id": "..."}
}
```

> 모든 인과·전망 표현에 근거(모델 입력 변수 / 농업 월보 / 기상특보 / web_search)가 함께 명시되도록 system prompt에서 강제됩니다. 막연한 단정 ("공급 감소", "수요 위축")은 금지.

### JSON 파싱 실패 시
`_parse_error: true` 와 `_raw` 필드가 있으면 GPT 응답이 JSON 표준을 위반한 것입니다. `explain.py` 에 자동 복구 fallback(문자열 안의 raw newline → `\n` escape)이 들어 있어 대부분 자동 처리되지만, 그래도 실패하면 `_raw` 의 원문을 직접 확인하세요.

---

## 4. 폴더 구조

```
xai_explainer/
├── README.md                       이 파일
├── requirements.txt
├── .gitignore                      __pycache__, outputs/*.json 등 제외
├── .env.example                    .env 로 복사 후 키 입력
├── .env                            ← (배포본에는 포함되지 않음. 사용자가 직접 생성)
├── run_xai.py                      진입점
├── config.py                       경로·키 로드
├── stn_code_map.py                 산지명 → 기상특보 광역 stnId
├── explain.py                      OpenAI Responses API + web_search
├── prompt_builder.py               System/User 프롬프트 조립
├── data_loaders/
│   ├── forecasts.py                inputs/forecast_*.parquet 로드
│   ├── context_summary.py          365일 → 요약 통계 텍스트
│   ├── weather_warn.py             기상특보 API 호출 (1주일치)
│   └── reports.py                  agri_report PDF 텍스트 추출
├── inputs/                         ★ 모든 입력 자료 동봉
│   ├── forecast_10day.parquet      Chronos2 10일 분위수 (★ 현재 mock — 아래 §5 참고)
│   ├── forecast_3day.parquet       TimesFM 3일 점예측 (★ 현재 mock)
│   ├── full_baseline.parquet       46품목 × 일별 365+ 일 raw
│   ├── meta_baseline.json          item_station_map 등
│   ├── dataset_descriptions/
│   │   ├── PROBABILISTIC.md        변수 사전 — 프롬프트에 첨부됨
│   │   └── POINT.md
│   └── agri_report/
│       ├── F202605.pdf             농넷 월보 (과일)
│       ├── FV202605.pdf            농넷 월보 (과채)
│       └── VC202605.pdf            농넷 월보 (채소)
├── outputs/                        실행 결과 저장 (실행 시 explanations_YYYYMMDD.json 생성)
└── scripts/
    └── refresh_forecasts.py        두 모델 재실행해 inputs/forecast_*.parquet 갱신
```

---

## 5. 입력 자료 명세

| 파일 | 형식 | 컬럼 | 상태 |
|---|---|---|---|
| `inputs/forecast_10day.parquet` | parquet | `item_id, timestamp, 0.1, 0.2, ..., 0.9, mean` | **★ 현재 mock 데이터** — 실제 예측값으로 교체 필요 |
| `inputs/forecast_3day.parquet`  | parquet | `item_id, timestamp, y_pred` | **★ 현재 mock 데이터** — 실제 예측값으로 교체 필요 |
| `inputs/full_baseline.parquet`  | parquet | `item_id, timestamp, target, + covariates 12종` | 원본 동봉 (2026-02-28까지) |
| `inputs/meta_baseline.json`     | json    | `known_covariates, past_covariates, item_station_map` | 원본 동봉 |
| `inputs/agri_report/*.pdf`      | pdf     | 농넷 농업관측 월보 (2026년 5월호) | 신규 호 추가 시 같은 폴더에 drop |

### ★ forecast parquet을 실제 예측값으로 교체하는 방법

현재 동봉된 `inputs/forecast_10day.parquet` 와 `inputs/forecast_3day.parquet` 는 파이프라인 검증용 **mock 데이터**입니다 (`full_baseline.parquet` 의 마지막 30일 통계 기반 임의 생성). 운영에 사용하려면 실제 두 모델의 예측값으로 반드시 교체해야 합니다.

교체 방법은 두 가지:

**(A) `scripts/refresh_forecasts.py` 사용** — `autogluon.timeseries` + `timesfm` 환경이 있는 PC에서 한 번 실행:
```bash
python scripts/refresh_forecasts.py
```
이 스크립트는 `final_handoff/PROBABILISTIC_FORECAST/` 와 `POINT_FORECAST/` 의 모델·데이터에 접근하여 forecast 두 parquet을 생성·덮어씁니다.

**(B) 외부에서 생성한 parquet을 직접 복사**:
- `forecast_10day.parquet`: 컬럼 `item_id, timestamp, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, mean` (46품목 × 10일)
- `forecast_3day.parquet`: 컬럼 `item_id, timestamp, y_pred` (46품목 × 3일)

---

## 6. 데이터 흐름

```
1. forecasts (parquet)  ──┐
2. full_baseline (365일) ─┤
3. WthrWrnInfoService   ──┼──► prompt_builder ──► OpenAI Responses API
4. agri_report (PDF)    ──┘                              │ web_search
5. dataset_descriptions ─► system prompt                 ▼
                                                outputs/*.json
```

---

## 7. 비용·성능 기준선

| 설정 | 회당 비용 | 소요 시간 |
|---|---|---|
| gpt-4o + web_search 활성화 (기본) | $2~4 | 4~12분 |
| gpt-4o + web_search 비활성화 (`--no-web-search`) | $1.5~2.5 | 2~6분 |
| gpt-4o-mini (`GPT_MODEL=gpt-4o-mini` 환경변수) | ~$0.05 | 1~2분 (설명 품질 저하 주의) |

- 출력이 두 문단(요약 1~2문장 + 설명 2문단)으로 간결해져 이전 12~18문장 narrative 대비 출력 토큰이 줄었습니다.
- OpenAI는 system prompt 자동 캐싱을 지원하므로 같은 날 반복 호출 시 비용이 더 줄어듭니다.
- LLM은 web_search 결과의 기간·주제 검증을 system prompt에서 강제 받으며, 검증 미통과 시 `web_sources`가 빈 배열로 남고 explanation에서도 인용하지 않습니다 — 무관한 외부 데이터로 인한 설명 오염 방지.

---

## 8. 트러블슈팅

| 증상 | 원인/조치 |
|---|---|
| `FileNotFoundError: forecast_10day.parquet` | `inputs/` 에 사전 캐싱된 파일이 없음. 사용자 측에서 `scripts/refresh_forecasts.py` 실행 후 산출물 동봉 필요. |
| 특보 API 응답 비어 있음 | 키 만료 / 요청 한도 / **해당 광역의 최근 1주일간 무특보**. 무특보의 경우 explanation에 "지난 1주일간 발효된 기상특보 없음" 이 한 문장으로 자동 반영됨. API 자체 실패는 `--skip-warnings` 로 우회 가능. 1주일 window는 `data_loaders/weather_warn.py` 의 `fetch_warnings_by_item(window_days_back=7)` 에서 조정 가능. |
| PDF 한글 깨짐 | `pypdf` 추출 한계. `pip install pdfplumber` 후 `data_loaders/reports.py` 의 추출 함수를 pdfplumber 로 교체. |
| `_parse_error: true` | GPT 응답이 JSON 형식이 아님. narrative 안의 raw newline은 `explain.py` 의 fallback 으로 자동 복구됨. 그래도 실패하면 `_raw` 의 원문 확인 후 system prompt 의 JSON 강제 지시를 더 강화. |
| `RuntimeError: .env에 ... 키가 비어 있음` | `.env.example` 참고해 키 입력. |
| OpenAI rate limit / web_search 비용 폭주 | `--no-web-search` 또는 `--limit N` 으로 제한. |
| `forecast_summary` / `forecast_explanation` 이 비거나 어색 | 모델이 가이드를 무시한 경우. `prompt_builder.py` 의 "출력 형식" 섹션의 예시 형식을 더 강하게 표현하거나 `GPT_MODEL=gpt-4o` 권장 (mini 모델은 가이드 준수도 낮음). |
| `web_sources` 가 항상 빈 배열 | LLM이 기간/주제 검증에서 매번 탈락시키는 중. 정상 동작일 수 있음 (관련 최신 뉴스가 실제로 없는 경우). 강제로 늘리고 싶으면 system prompt의 web_search 가이드를 완화할 수 있으나 권장하지 않음. |

---

## 9. 예측 결과 갱신 (사용자 운영용)

`scripts/refresh_forecasts.py` 는 `final_handoff/PROBABILISTIC_FORECAST/`, `POINT_FORECAST/` 와 모델 파일이 같이 있을 때만 동작합니다. 두 모델을 새로 돌려 `inputs/forecast_*.parquet` 를 덮어씁니다.

```bash
# autogluon.timeseries / timesfm 환경에서
python scripts/refresh_forecasts.py
```

팀원에게 보낼 때는 사용자 측에서 한 번 실행해 갱신된 parquet 두 개를 동봉하면 됩니다.

---

## 10. 참고

- 변수 사전: `inputs/dataset_descriptions/PROBABILISTIC.md`, `POINT.md`
- 기상특보 API 명세: 기상청 공공데이터포털 "기상특보 조회서비스" Open API
- OpenAI Responses API + web_search 도구: <https://platform.openai.com/docs/guides/tools-web-search>

---

## 11. 변경 이력 (주요)

### 최신 — 근거 명시 강제 + 기상특보 1주일 확장
- **근거 명시 규칙 추가**: forecast_explanation의 인과·전망·해석성 표현("공급 감소", "수요 위축", "가격 압박" 등)에 반드시 해당 주장의 출처(모델 입력 변수 / 농업 월보 / 기상특보 / web_search 결과 중 하나)를 문장 안에 자연스럽게 함께 명시하도록 system prompt에 강제 규칙 추가. 글의 자연스러움을 해치지 않는 균형도 함께 가이드.
- **기상특보 1주일치로 확장**: `fetch_warnings_by_item()` 기본 window를 D-3 → **D-7**로 확장. `format_warnings()`는 1주일간 발효된 모든 특보를 발표시각(tmFc) 최신순으로 정렬해 LLM에 전달하며, 각 특보의 발표 일시(`YYYY-MM-DD HH:MM`)를 함께 표기. system prompt에 1주일치 특보 활용 가이드(빈도 짚기, 일자·광역·종류 함께 인용) 추가.

### XAI 목적 재정의 및 출력 단순화
- **XAI 역할 재정의**: 기존 "예측값을 뒷받침하는 분석" → **"모델 예측을 결론으로 받아들이고, 그 결론이 왜 나왔는지 사후 설명"**. system prompt 전면 재작성.
- **출력 단순화 (스키마 슬림화)**: `narrative`, `key_drivers`, `scenario_analysis`, `risks`, `cross_check`, `weather_warning_status`, `direction`, `horizon_summary`, `confidence_interval_reading`, `one_line_summary` 등 다수 필드 제거. 다음 두 핵심 필드만 유지:
  - **`forecast_summary`** — 1~2문장, 모델 예측(3일/10일) 요약
  - **`forecast_explanation`** — 2문단, 위 결론의 사후 설명
- **수치 인용 강제 폐지**: 이전 "모든 문장에 수치 인용 의무" 규칙이 글을 난잡하게 만들어 제거. 수치는 근거가 될 때만 자연스럽게 인용.
- **사후 추론 어조 채택**: "~로 보입니다", "~로 추정됩니다" 같은 추정 표현 사용. 단정 금지.
- **web_search 검증 필수화**: 사용 시 반드시 두 가지 검증을 거치고 둘 다 통과한 결과만 인용:
  1. **기간 검증** — 최근 30일 이내 발행 자료인가
  2. **주제 부합성 검증** — 한국 농산물 도매가격·수급·작황·정책·기상 재해에 직접 관련되는가
  통과 결과만 `web_sources`에 추가하고 `verified_date_range`, `topic_relevance` 필드에 검증 근거 명시. 통과 결과가 없으면 빈 배열로 두고 본문에서도 인용 금지.

### 이전 변경
- **기상특보 빈 응답 처리**: 무특보일 때 명시적 문구 ("현재 시점 기준 해당 광역에 발효 중인 활성·예비 기상특보 없음") 출력 + explanation에 한 문장으로 반영.
- **JSON 파싱 견고화**: GPT가 긴 문자열 안에 raw newline을 넣어 JSON 표준을 위반해도 `explain.py` 의 fallback 으로 자동 escape 후 재파싱.
- **JSON escape 규칙 명시**: system prompt 응답 형식에 "문자열 값 내 줄바꿈은 `\n` escape" 규칙 추가.
