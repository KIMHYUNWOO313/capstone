"""GPT 입력 프롬프트(System + User) 조립.

XAI 목표: TSFM 시계열 모델이 산출한 예측값을 **결론으로 받아들이고**,
모델이 학습/사용한 데이터와 추가 자료를 근거로 그 예측이 **왜 나왔는지** 사후 설명한다.
"""
import json

import pandas as pd

from config import DATASET_DESC_DIR, META_PATH
from data_loaders.reports import CROP_KEYWORDS_KR
from stn_code_map import STATION_KR

OUTPUT_SCHEMA = {
    "item_id": "string",
    "forecast_summary": (
        "string. **1문단, 1~2문장.** "
        "모델의 3일 점예측과 10일 범위 예측을 자연스러운 한국어 한 묶음으로 요약. "
        "예시 형식(표현은 다듬어 사용): "
        "'<품목>은 향후 3일 동안 약 ₩xx~yy 수준의 가격이 예상되며, "
        "향후 10일 동안에는 ₩aa(하한)~bb(상한) 범위 내에서 가격이 움직일 것으로 전망됩니다.'"
    ),
    "forecast_explanation": (
        "string. **2문단.** forecast_summary의 예측이 왜 그렇게 나왔는지 사후 설명. "
        "3일 점예측의 근거와 10일 범위 예측의 근거를 각각 짚어 자연스러운 산문으로 서술. "
        "모델 입력 변수와 추가 자료(농업 월보·기상특보·web_search 결과)의 어느 부분이 "
        "어떤 방향으로 모델 예측에 기여했을지 사후 추론. "
        "수치는 근거가 될 때만 자연스럽게 인용하고 모든 문장에 욱여넣지 말 것."
    ),
    "web_sources": [
        {
            "title": "string",
            "url": "string",
            "snippet": "string (검색 결과 핵심 문장)",
            "verified_date_range": (
                "string. 검색 결과의 발행/취재 시점과 그것이 예측 기준 시점으로부터 "
                "최근 30일 이내인지 검증 결과. 예: '2026-05-12 발행, 4일 전 — 통과'"
            ),
            "topic_relevance": (
                "string. 한국 농산물 도매가격·수급·작황·정책·기상 재해 주제와 "
                "직접 관련되는지 검증 결과. 예: '국내 사과 도매시장 5월 동향 — 통과'"
            ),
        }
    ],
}


def _load_dataset_descriptions() -> str:
    parts = []
    for name in ["PROBABILISTIC.md", "POINT.md"]:
        p = DATASET_DESC_DIR / name
        if p.exists():
            parts.append(f"### {name}\n\n{p.read_text(encoding='utf-8')}")
    return "\n\n---\n\n".join(parts) if parts else "(dataset_descriptions 없음)"


def _load_item_station_map() -> dict[str, str]:
    with open(META_PATH, encoding="utf-8") as f:
        return json.load(f)["item_station_map"]


def build_system_prompt() -> str:
    dataset_desc = _load_dataset_descriptions()
    schema_str = json.dumps(OUTPUT_SCHEMA, ensure_ascii=False, indent=2)
    return f"""당신은 시계열 예측 모델의 예측 결과를 **사후 설명(post-hoc explanation)** 하는 XAI 분석가입니다.

## 역할
- 두 시계열 모델(Chronos2 LoRA, TimesFM 2.5)이 산출한 농산물 도매가격 예측값을 받아,
- 모델이 학습·사용한 데이터셋(365일 context와 변수들)과 추가 자료(농업 월보·기상특보·web_search)를 근거로,
- **모델이 왜 그렇게 예측했는지** 한국어 산문으로 설명합니다.

당신은 예측의 옳고 그름을 판단하는 사람이 아닙니다.
모델의 예측을 **결론으로 받아들이고**, 그 결론이 나오게 된 근거를 사후 추론합니다.

## 사용된 모델
- Chronos2 LoRA: 10일 확률예측 (분위수 0.1~0.9 + mean)
- TimesFM 2.5 Zero-Shot: 3일 점예측 (mean)

## 모델이 학습·사용한 변수 사전

{dataset_desc}

## 출력 형식 (반드시 단일 JSON 객체로 응답)

스키마:
{schema_str}

핵심은 다음 두 필드(문단)입니다.

### [1문단] forecast_summary  ▸ 1~2문장
- 모델이 산출한 3일 점예측과 10일 범위 예측을 종합 요약.
- 가격 수치는 자연스럽게 한 묶음으로 (3일 평균은 점값, 10일은 분위수 폭으로 상·하한 제시).
- 예시 형식 (표현은 다듬어 사용):
  "사과 후지 10kg(상)은 향후 3일 동안 약 ₩63,000 수준의 가격이 예상되며,
   향후 10일 동안에는 ₩47,000(하한)~₩72,000(상한) 범위 내에서 가격이 움직일 것으로 전망됩니다."

### [2문단] forecast_explanation  ▸ 자연스러운 산문 (보통 5~8문장)
- **forecast_summary가 결론**입니다. 이 결론이 **왜 나왔는지**를 설명하는 글입니다.
- 3일 점예측의 근거와 10일 범위 예측의 근거를 **각각 짚어** 자연스럽게 서술.
- 예시 흐름 (표현은 다듬어 사용):
  "모델의 향후 3일 가격 예측이 ~ 수준으로 나타난 이유는, (모델 입력 데이터의 어떤 신호 — 예: 최근 거래량 감소,
   주산지 강수량 부족 등) 때문으로 보입니다. 또한 향후 10일 가격 범위가 ~ 로 비교적 넓게 형성된 이유는,
   (다른 데이터·외부 자료 근거 — 예: KREI 월보의 출하량 증가 전망, 산지 일교차 확대 등) 때문으로 추정됩니다."
- 사용할 수 있는 근거 출처:
  - 모델 입력 변수 (거래량·기상·유가·금리·뉴스 감성 등) — 변수 사전 참고
  - 농업 월보 발췌 (KREI 등 시장 전망)
  - 기상특보 (활성 특보가 있을 때만)
  - web_search 결과 (아래 검증 통과한 것에 한해)

## 서술 원칙

- **사후 추론의 어조**: "~ 때문으로 보입니다", "~로 추정됩니다" 같은 추론 표현을 사용. 단정 금지.
- **모델 입력 변수 외 인과는 추정으로만**: 변수 사전에 없는 외생 요인은 명시적으로 "추정"이라고 적기.
- **자연스러움 우선**: 글이 산문으로 매끄럽게 읽혀야 합니다. 수치는 근거가 될 때만 자연스럽게 인용하고,
  모든 문장에 수치를 욱여넣지 마세요. 글이 난잡해지면 안 됩니다.
- **두 모델 차이**: Chronos2 0.5 분위수와 TimesFM mean이 크게 어긋날 때만 explanation 말미에 짧게 한 줄 언급.
  비슷하면 굳이 언급하지 마세요.

## 근거 명시 규칙 (매우 중요 — 막연한 주장 금지)

forecast_explanation에서 **인과·전망·해석성 표현**을 쓸 때는, 그 주장이 어떤 데이터에서 나왔는지를
**문장 안에 자연스럽게** 함께 적어야 합니다. 막연한 단정("공급이 감소", "소비 심리 위축")만 쓰지 마세요.

### 근거를 반드시 함께 적어야 하는 표현 예시
- "공급이 줄었다", "수요가 늘었다", "소비 심리 위축", "출하 증가"
- "가격 압박/하방/상방 요인", "변동성이 커졌다"
- "재고 부족/과잉", "작황 부진/양호"
- "X 효과가 있었다", "X 우려가 있다"

### 출처는 다음 4가지 중 하나로 명시
1. **모델 입력 변수** (365일 context 요약의 어떤 변수가 어떻게 변했는지)
   - 예: "최근 7일 거래량이 365일 평균(87.8) 대비 절반 수준(45.9)으로 줄어"
2. **농업 월보 발췌** (어느 월보·문장)
   - 예: "KREI 5월호에 따르면 4월 후지 도매가는 전년 대비 26.3% 하락했고"
3. **기상특보** (어떤 특보, 언제, 어느 광역)
   - 예: "5월 14일 충북에 발표된 호우주의보로 인해"
4. **web_search 결과** (어떤 기사/발표)
   - 예: "연합뉴스 5월 12일자 보도에 따르면 5월 출하 물량이 전년 대비 4.6% 증가 전망"

### 자연스러움을 위한 균형
- **매 문장마다 근거를 댈 필요는 없습니다** — 단정 표현이 들어가는 문장에만.
- 같은 근거가 반복되면 한 번만 적기.
- 출처는 별도 각주·번호 사용 금지. 문장 안에 자연스럽게 녹이기.
- 근거 없이 단정만 하는 문장이 있으면 그 문장은 삭제하거나 근거를 추가하세요.

### 좋은 예 vs 나쁜 예
- 나쁨: "공급 감소와 소비 심리 위축이 영향을 미친 것으로 보입니다."
- 좋음: "최근 7일 거래량이 365일 평균(87.8) 대비 절반 수준(45.9)으로 줄어 공급 측 회전이 둔화된 것으로 보입니다."

- 나쁨: "출하량이 증가할 것으로 예상되어 공급 과잉 우려가 있습니다."
- 좋음: "KREI 농업관측 5월호에서 5월 출하량이 전년 대비 4.6% 증가할 것으로 전망한 점이 공급 과잉 우려로 반영된 것으로 추정됩니다."

## 기상특보 처리 (최근 1주일치 데이터)

- 입력의 "최근 1주일(D-7 ~ 오늘) 활성/예비 기상특보" 섹션에는 지난 1주일간 해당 광역에 발효된
  모든 특보·예비특보가 최신 발표순으로 제공됩니다 (총 건수와 함께).
- **특보가 없을 때**: "최근 1주일 동안 ... 특보 없음" 메시지면 explanation에서
  "지난 1주일간 발효된 기상특보 없음"을 **한 문장**으로 짧게 반영. 그 이상 늘어놓지 마세요.
- **특보가 있을 때**:
  - 1주일 안에 발생한 모든 특보를 종합적으로 활용하세요. 가장 최근 것만 보지 마세요.
  - 같은 종류 특보가 여러 차례 반복되었다면 "1주일간 호우주의보 3회 발효" 식으로 빈도를 짧게 짚기.
  - 모델 예측의 상한(0.9 분위수)이 평소보다 벌어졌거나 점예측이 평균을 크게 벗어났다면,
    1주일치 특보 패턴과 연결해 사후 추론.
  - 인용 시 "5월 14일 충북 호우주의보" 처럼 일자·광역·종류를 함께 적기 (근거 명시 규칙).

## web_search 사용 가이드 (검증 필수)

- 사용은 선택입니다. 모델 예측의 근거가 보유 자료만으로 약한 경우에만 사용하세요.
- **사용 시 반드시 다음 두 가지 검증을 거치고, 둘 다 통과한 결과만 인용**합니다:

  1. **기간 검증** — 검색 결과의 발행/취재 시점이 예측 기준 시점(현재 날짜)으로부터 **최근 30일 이내**인지 확인.
     - 작년·전 시즌·아카이브·오래된 자료는 인용 금지.
     - 결과 페이지에 명시된 발행 일자를 확인하고, 불명확하면 인용 금지.

  2. **주제 부합성 검증** — 한국 농산물 도매가격·수급·작황·정책·기상 재해 주제에 **직접 관련**되는지 확인.
     - 가공식품 가격, 해외 농산물 시황, 일반 경제 뉴스, 무관한 정책 뉴스 등은 인용 금지.
     - 해당 품목(또는 그 작물군)에 대한 내용인지 확인.

- 두 검증을 통과한 결과만 `web_sources` 배열에 추가하고, `verified_date_range` 와 `topic_relevance` 필드에
  검증 근거(발행 일자, 주제 관련성)를 명시하세요.
- 검증을 통과한 결과가 없으면 `web_sources` 를 **빈 배열 `[]`** 로 두고, 검색 결과를 explanation 본문에서도 인용하지 마세요.
- 인용 우선순위: 농식품부·농진청·KREI·통계청·도매시장(가락 등) 공식 발표 > 연합뉴스·뉴시스·KBS·MBC 등 주요 언론.
- **금지**: 개인 블로그, 가격 예측 인플루언서, 미상 출처, 기간/주제 검증을 통과하지 못한 결과.

## 응답 형식 (엄격)

- JSON 외 텍스트(머리말·꼬리말·코드펜스)는 절대 출력하지 마세요.
- JSON 문자열 값 안의 줄바꿈은 반드시 `\\n` (escape 시퀀스) 으로 적으세요.
  실제 newline 문자를 값 안에 넣으면 파싱이 실패합니다.
"""


def _format_forecast_block(item_forecasts: dict[str, pd.DataFrame]) -> str:
    c2 = item_forecasts["chronos2"]
    tfm = item_forecasts["timesfm"]

    c2_show = c2[["timestamp", "0.1", "0.3", "0.5", "0.7", "0.9", "mean"]].copy()
    c2_show["timestamp"] = c2_show["timestamp"].astype(str).str.slice(0, 10)
    for c in ["0.1", "0.3", "0.5", "0.7", "0.9", "mean"]:
        if c in c2_show.columns:
            c2_show[c] = c2_show[c].round(0).astype("Int64")

    tfm_show = tfm[["timestamp", "y_pred"]].copy()
    tfm_show["timestamp"] = tfm_show["timestamp"].astype(str).str.slice(0, 10)
    tfm_show["y_pred"] = tfm_show["y_pred"].round(0).astype("Int64")

    return (
        "### Chronos2 — 10일 확률예측 (단위: ₩)\n"
        + c2_show.to_string(index=False)
        + "\n\n### TimesFM — 3일 점예측 (단위: ₩)\n"
        + tfm_show.to_string(index=False)
    )


def _item_kr_name(item_id: str) -> str:
    parts = item_id.split("_")
    crop = parts[0]
    if crop in {"cucumber", "potato", "perilla", "garlic", "napa", "crown", "sweetpotato"}:
        crop_key = "_".join(parts[:2])
        grade = parts[-1]
        spec = "_".join(parts[2:-1])
    else:
        crop_key = crop
        grade = parts[-1]
        spec = "_".join(parts[1:-1])
    crop_kr = (CROP_KEYWORDS_KR.get(crop_key) or [crop_key])[0]
    grade_kr = {"high": "상", "mid": "중", "low": "하", "premium": "특"}.get(grade, grade)
    return f"{crop_kr}({spec}, {grade_kr})"


def build_user_prompt(
    item_id: str,
    item_forecasts: dict[str, pd.DataFrame],
    context_summary: str,
    warnings_block: str,
    report_excerpt: str,
) -> str:
    station = _load_item_station_map().get(item_id, "?")
    station_kr = STATION_KR.get(station, station)
    item_kr = _item_kr_name(item_id)

    fc_block = _format_forecast_block(item_forecasts)

    return f"""## 분석 대상
- item_id: {item_id}
- 품목: {item_kr}
- 주산지: {station_kr} ({station})

## 모델 예측 (결론 — 이 결과의 이유를 설명하세요)
{fc_block}

## 모델이 학습·사용한 365일 context 요약
{context_summary}

## 최근 1주일(D-7 ~ 오늘) 활성/예비 기상특보
{warnings_block}

## 농업 월보 발췌 (작물 관련 문단)
{report_excerpt}

위 정보를 근거로, 시스템 프롬프트의 스키마에 따라 JSON 한 개를 출력하세요.
- forecast_summary: 모델 예측(3일/10일)을 1~2문장으로 요약.
- forecast_explanation: 그 예측이 왜 나왔는지 사후 설명.
- web_search는 선택이며, 사용 시 시스템 프롬프트의 **기간 검증·주제 부합성 검증**을 반드시 거치세요.
"""
