# -*- coding: utf-8 -*-
"""OpenAI GPT로 예측 근거·한계·모니터링 포인트 설명."""
from __future__ import annotations

import importlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pandas as pd
from django.conf import settings

from .agri_data import (
    covariate_and_news_context_for_explain,
    format_weather_economic_block_for_explain,
    item_option_meta,
    summarize_baseline_context_for_explain,
)
from .agri_store import get_latest_predict_batch


def _sanitize_explanation_output(text: str) -> str:
    """모델이 마크다운 강조를 쓴 경우 화면에 ** 등이 보이지 않게 정리."""
    if not text:
        return text
    text = text.replace("**", "")
    text = text.replace("__", "")
    text = text.replace("*", "")
    lines: list[str] = []
    for line in text.splitlines():
        s = line.lstrip()
        if s.startswith("#### "):
            line = line.replace("#### ", "", 1)
        elif s.startswith("### "):
            line = line.replace("### ", "", 1)
        elif s.startswith("## "):
            line = line.replace("## ", "", 1)
        elif s.startswith("# "):
            line = line.replace("# ", "", 1)
        lines.append(line)
    return "\n".join(lines).strip()


def _load_xai_prompt_builder():
    base_dir = Path(getattr(settings, "BASE_DIR", "."))
    xai_dir = base_dir / "xai" / "xai" / "xai_final" / "xai_explainer"
    prompt_path = xai_dir / "prompt_builder.py"
    if not xai_dir.is_dir() or not prompt_path.exists():
        return None
    xai_dir_str = str(xai_dir)
    if xai_dir_str not in sys.path:
        sys.path.insert(0, xai_dir_str)
    old_config_module = sys.modules.pop("config", None)
    old_prompt_builder = sys.modules.pop("prompt_builder", None)
    try:
        spec = importlib.util.spec_from_file_location(
            "agri_xai_prompt_builder",
            prompt_path,
        )
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        sys.modules["agri_xai_prompt_builder"] = module
        spec.loader.exec_module(module)
        if not hasattr(module, "build_system_prompt") or not hasattr(module, "build_user_prompt"):
            return None
        return module
    except Exception as exc:
        print(f"[agri_gpt] xai_explainer prompt_builder load failed: {exc.__class__.__name__}: {exc}")
        return None
    finally:
        sys.modules.pop("prompt_builder", None)
        if old_config_module is not None:
            sys.modules["config"] = old_config_module
        if old_prompt_builder is not None:
            sys.modules["prompt_builder"] = old_prompt_builder


def _load_xai_module(filename: str, module_key: str):
    base_dir = Path(getattr(settings, "BASE_DIR", "."))
    xai_dir = base_dir / "xai" / "xai" / "xai_final" / "xai_explainer"
    module_path = xai_dir / filename
    if not module_path.is_file():
        return None
    xai_dir_str = str(xai_dir)
    if xai_dir_str not in sys.path:
        sys.path.insert(0, xai_dir_str)
    old_config_module = sys.modules.pop("config", None)
    try:
        spec = importlib.util.spec_from_file_location(module_key, module_path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_key] = module
        spec.loader.exec_module(module)
        return module
    except Exception as exc:
        print(f"[agri_gpt] xai module load failed ({filename}): {exc.__class__.__name__}: {exc}")
        return None
    finally:
        if old_config_module is not None:
            sys.modules["config"] = old_config_module


def _crop_key_from_item_id(item_id: str) -> str:
    parts = item_id.split("_")
    crop = parts[0]
    if crop in {"cucumber", "potato", "perilla", "garlic", "napa", "crown", "sweetpotato"}:
        return "_".join(parts[:2])
    return crop


def _build_xai_warnings_block(item_id: str, cov: dict[str, Any]) -> str:
    cov_block = format_weather_economic_block_for_explain(cov)
    warn_mod = _load_xai_module("data_loaders/weather_warn.py", "agri_xai_weather_warn")
    if warn_mod is not None:
        try:
            api_key = getattr(warn_mod, "WARN_API_KEY", "") or ""
            if api_key.strip():
                by_item = warn_mod.fetch_warnings_by_item()
                warn_text = warn_mod.format_warnings(by_item.get(item_id, []))
                return cov_block + "\n\n### 최근 1주일 기상특보\n" + warn_text
        except Exception as exc:
            print(f"[agri_gpt] weather warning fetch failed: {exc.__class__.__name__}: {exc}")
    return cov_block + (
        "\n\n### 최근 1주일 기상특보\n"
        "별도 기상특보 API 연결 없음. 위 기상·경제 지표와 최근 가격 흐름을 외부 요인 근거로 활용하세요."
    )


def _build_xai_report_excerpt(item_id: str, cov: dict[str, Any]) -> str:
    reports_mod = _load_xai_module("data_loaders/reports.py", "agri_xai_reports")
    if reports_mod is not None:
        try:
            reports = reports_mod.load_all_reports()
            crop_key = _crop_key_from_item_id(item_id)
            excerpt = reports_mod.relevant_excerpts_for_crop(reports, crop_key)
            if excerpt and excerpt != "(관련 월보 발췌 없음)":
                return excerpt
        except Exception as exc:
            print(f"[agri_gpt] agri report load failed: {exc.__class__.__name__}: {exc}")

    news_txt = cov.get("참고_뉴스_글")
    if news_txt:
        return str(news_txt)[:2000]

    return (
        "농업 월보 PDF 발췌는 연결되어 있지 않습니다. "
        "대신 위 365일 context 요약·최근 2주 기상·경제 지표·가격 흐름을 근거로 설명하세요. "
        "'기상·경제 자료 없음'이라고 쓰지 마세요."
    )


def _safe_num(v: Any) -> float | None:
    try:
        n = float(v)
        return n if n == n else None
    except Exception:
        return None


def _interp(a: float | None, b: float | None, t: float) -> float | None:
    if a is None and b is None:
        return None
    if a is None:
        return b
    if b is None:
        return a
    return a + (b - a) * t


def _build_item_forecasts_for_xai(item_id: str, chart_summary: dict[str, Any]) -> dict[str, pd.DataFrame]:
    prob = get_latest_predict_batch(item_id, model_name="probabilistic")
    point = get_latest_predict_batch(item_id, model_name="point")

    prob_rows: list[dict[str, Any]] = []
    for p in list((prob or {}).get("points") or []):
        p10 = _safe_num(p.get("p10"))
        p50 = _safe_num(p.get("p50", p.get("pred_krw")))
        p90 = _safe_num(p.get("p90"))
        row = {
            "timestamp": p.get("date"),
            "0.1": p10,
            "0.3": _interp(p10, p50, 0.5),
            "0.5": p50,
            "0.7": _interp(p50, p90, 0.5),
            "0.9": p90,
            "mean": _safe_num(p.get("pred_krw", p.get("p50"))),
        }
        if row["timestamp"]:
            prob_rows.append(row)

    if not prob_rows:
        for p in (chart_summary.get("points") or []):
            if p.get("kind") != "predict":
                continue
            p10 = _safe_num(p.get("p10", p.get("p1")))
            p50 = _safe_num(p.get("value"))
            p90 = _safe_num(p.get("p90", p.get("p99")))
            row = {
                "timestamp": p.get("date"),
                "0.1": p10,
                "0.3": _interp(p10, p50, 0.5),
                "0.5": p50,
                "0.7": _interp(p50, p90, 0.5),
                "0.9": p90,
                "mean": p50,
            }
            if row["timestamp"]:
                prob_rows.append(row)

    point_rows: list[dict[str, Any]] = []
    for p in list((point or {}).get("points") or []):
        ts = p.get("date")
        pred = _safe_num(p.get("y_pred", p.get("pred_krw")))
        if ts and pred is not None:
            point_rows.append({"timestamp": ts, "y_pred": pred})
    if not point_rows:
        fallback_predicts = [p for p in (chart_summary.get("points") or []) if p.get("kind") == "predict"][:3]
        for p in fallback_predicts:
            ts = p.get("date")
            pred = _safe_num(p.get("value"))
            if ts and pred is not None:
                point_rows.append({"timestamp": ts, "y_pred": pred})

    chronos_df = pd.DataFrame(
        prob_rows
        or [{"timestamp": "", "0.1": None, "0.3": None, "0.5": None, "0.7": None, "0.9": None, "mean": None}]
    )
    timesfm_df = pd.DataFrame(point_rows or [{"timestamp": "", "y_pred": None}])
    for col in ["0.1", "0.3", "0.5", "0.7", "0.9", "mean"]:
        if col in chronos_df.columns:
            chronos_df[col] = pd.to_numeric(chronos_df[col], errors="coerce")
    if "y_pred" in timesfm_df.columns:
        timesfm_df["y_pred"] = pd.to_numeric(timesfm_df["y_pred"], errors="coerce")
    return {
        "chronos2": chronos_df,
        "timesfm": timesfm_df,
    }


def _extract_response_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if text:
        return text
    chunks: list[str] = []
    for item in getattr(response, "output", []) or []:
        for c in getattr(item, "content", []) or []:
            t = getattr(c, "text", None)
            if t:
                chunks.append(t)
    return "\n".join(chunks).strip()


def _parse_response_json(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    try:
        return json.loads(s)
    except Exception:
        pass
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _series_stats(chart_summary: dict) -> dict[str, Any]:
    """차트·수치 요약(10일 예측 증감 중심). 키는 비전문가도 읽기 쉬운 한글 위주."""
    pts = chart_summary.get("points") or []
    actual = [p for p in pts if p.get("kind") == "actual"]
    predict = [p for p in pts if p.get("kind") == "predict"]

    item_id = str(chart_summary.get("item_id") or "")
    try:
        item_meta = item_option_meta(item_id)
    except Exception:
        item_meta = {"label": item_id, "crop": chart_summary.get("crop_ko"), "unit": "", "grade": ""}

    out: dict[str, Any] = {
        "품목_이름": chart_summary.get("crop_ko"),
        "선택_품목_표시명": item_meta.get("label"),
        "선택한_작물": item_meta.get("crop"),
        "선택한_단위명": item_meta.get("unit"),
        "선택한_등급명": item_meta.get("grade"),
        "선택한_규격_코드": chart_summary.get("item_id"),
        "데이터에서_가장_늦은_날짜": chart_summary.get("last_csv_date"),
        "차트에_나온_실제가_일_수": len(actual),
        "차트에_나온_예측_일_수": len(predict),
    }

    if actual:
        vals = [float(p["value"]) for p in actual]
        out["최근_실제_가격_개요"] = {
            "이_구간_첫_날": actual[0]["date"],
            "이_구간_마지막_날": actual[-1]["date"],
            "이_구간_중_가장_낮은_가격_원": round(min(vals), 2),
            "이_구간_중_가장_높은_가격_원": round(max(vals), 2),
            "가장_최근에_확정된_날": actual[-1]["date"],
            "그날_가격_원": round(float(actual[-1]["value"]), 2),
        }

    if predict:
        pv = [float(p["value"]) for p in predict]
        out["앞으로_예측된_가격_개요"] = {
            "예측_첫_날": predict[0]["date"],
            "예측_마지막_날": predict[-1]["date"],
            "첫날_예상_가격_원": round(pv[0], 2),
            "마지막날_예상_가격_원": round(pv[-1], 2),
            "예측_기간_중_가장_낮은_가격_원": round(min(pv), 2),
            "예측_기간_중_가장_높은_가격_원": round(max(pv), 2),
            "예측_기간_평균_가격_원": round(sum(pv) / len(pv), 2),
        }
        if len(pv) >= 2:
            d0, d1 = pv[0], pv[-1]
            if d0 != 0:
                out["예측_첫날_대비_마지막날_변동_퍼센트"] = round(
                    (d1 - d0) / abs(d0) * 100.0, 3
                )
            eps = max(abs(d0) * 1e-6, 1e-9)
            if d1 > d0 + eps:
                out["예측_전체_가격_방향"] = "상승"
            elif d1 < d0 - eps:
                out["예측_전체_가격_방향"] = "하락"
            else:
                out["예측_전체_가격_방향"] = "횡보(소폭)"
            steps_up = sum(1 for i in range(1, len(pv)) if pv[i] >= pv[i - 1])
            out["예측_기간_하루씩_전날보다_오른_날의_비율"] = round(
                steps_up / (len(pv) - 1), 3
            )
        if actual:
            la = float(actual[-1]["value"])
            fp = float(predict[0]["value"])
            if la != 0:
                out["가장_최근_실제가_대비_첫_예측일_변동_퍼센트"] = round(
                    (fp - la) / abs(la) * 100.0, 3
                )

    return out


def _chart_points_for_user(pts: list[dict]) -> list[dict[str, Any]]:
    """차트 포인트 kind를 한글 구분으로 (영문 코드 노출 최소화)."""
    out: list[dict[str, Any]] = []
    for p in pts:
        k = p.get("kind")
        if k == "actual":
            구분 = "실제_거래가_또는_확정가"
        elif k == "predict":
            구분 = "모델_예측가"
        else:
            구분 = str(k)
        out.append(
            {
                "날짜": p.get("date"),
                "가격_원_근사": p.get("value"),
                "구분": 구분,
            }
        )
    return out


def _explain_result(expert: str, plain: str | None = None) -> dict[str, str]:
    expert_text = _sanitize_explanation_output(expert or "")
    plain_text = _sanitize_explanation_output(plain or "") if plain else ""
    if not plain_text:
        plain_text = expert_text
    return {"explanation": expert_text, "explanation_plain": plain_text}


def _compose_explanation_from_parsed(parsed: dict[str, Any]) -> tuple[str, str, str]:
    summary = str(parsed.get("forecast_summary") or "").strip()
    explanation = str(parsed.get("forecast_explanation") or "").strip()
    summary_plain = str(parsed.get("forecast_summary_plain") or "").strip()
    explanation_plain = str(parsed.get("forecast_explanation_plain") or "").strip()

    expert_parts = [part for part in (summary, explanation) if part]
    plain_parts = [
        part for part in (
            summary_plain or summary,
            explanation_plain or explanation,
        ) if part
    ]

    web_sources = parsed.get("web_sources") or []
    if isinstance(web_sources, list) and web_sources:
        previews = []
        for src in web_sources[:3]:
            if not isinstance(src, dict):
                continue
            title = str(src.get("title") or "").strip()
            url = str(src.get("url") or "").strip()
            if title and url:
                previews.append(f"{title} ({url})")
            elif title:
                previews.append(title)
        if previews:
            expert_parts.append("참고 출처: " + " ; ".join(previews))

    expert = _sanitize_explanation_output("\n\n".join(expert_parts))
    plain = _sanitize_explanation_output("\n\n".join(plain_parts))
    return expert, plain, explanation_plain


def _simplify_explanation_for_laypeople(client, model: str, expert_text: str) -> str:
    source = str(expert_text or "").strip()
    if not source:
        return ""
    prompt = (
        "아래 농산물 가격 예측 설명을 농부·시장 상인도 바로 이해할 수 있게 바꿔 주세요.\n"
        "내용(예측 방향, 이유, 영향 요인)은 그대로 두고 표현만 쉽게 하세요.\n"
        "금지: 분위수, 공변량, target, 시계열, LoRA, 모델명, 영어 전문용어.\n"
        "쉬운 말 예: 점예측→앞 3일 예상 가격, 구간 예측→앞 10일 예상 범위, "
        "변동성→가격 출렁임, 중심값→가장 그럴듯한 예상 가격.\n"
        "짧은 문장, 존댓말, 마크다운 금지.\n\n"
        f"원문:\n{source[:3500]}"
    )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You rewrite Korean agricultural price explanations for non-expert farmers and market sellers.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=1800,
        )
        return _sanitize_explanation_output((response.choices[0].message.content or "").strip())
    except Exception:
        return source


def explain_forecast(
    item_id: str,
    chart_summary: dict,
    batch_meta: dict | None,
) -> dict[str, str]:
    api_key = getattr(settings, "OPENAI_API_KEY", "") or ""
    if not api_key.strip():
        msg = (
            "OPENAI_API_KEY가 설정되지 않았습니다. `.env`에 키를 넣으면 "
            "GPT 기반 설명을 생성합니다.\n\n"
            f"품목: {item_id}\n"
            f"차트 포인트 수: {len(chart_summary.get('points', []))}"
        )
        return _explain_result(msg)

    from openai import AuthenticationError, OpenAI, OpenAIError

    client = OpenAI(api_key=api_key)
    model = getattr(settings, "OPENAI_EXPLAIN_MODEL", "gpt-4o-mini")

    pts = chart_summary.get("points", [])
    tail_raw = pts[-30:] if len(pts) > 30 else pts
    tail = _chart_points_for_user(tail_raw)
    stats = _series_stats(chart_summary)
    try:
        cov = covariate_and_news_context_for_explain(item_id)
    except Exception as exc:
        cov = {"자료_불러오기_실패": str(exc)}

    예측_설명_메타: dict[str, Any] | None = None
    if batch_meta:
        예측_설명_메타 = {
            "예측을_만든_기준_날_마지막_실제_관측일": batch_meta.get("origin_date"),
            "앞으로_몇_일을_예측했는지": batch_meta.get("horizon"),
        }

    user_payload = {
        "핵심_목표": (
            "(1) 10일 예측 가격으로 증감 방향을 판단하고 "
            "(2) 아래 기상·경제 지표 및(있을 경우) 뉴스 글로 "
            "증감 원인을 추정한다. 모두 추정이며 단정 금지."
        ),
        "가격_예측_요약": stats,
        "기상_경제_뉴스_자료": cov,
        "차트_최근_가격_추이": tail,
        "예측_실행_정보": 예측_설명_메타,
    }

    fallback_system = """역할: 농산물 가격 구간 예측(10일) 결과를 한국어로 설명한다.

출력 형식 규칙(엄수):
- 마크다운 금지: 별표 두 개, 별표 한 개, 해시(#), 백틱, 굵게 표시용 기호를 절대 쓰지 않는다.
- 제목이 필요하면 한 줄 끝에 콜론(:)만 쓰거나 번호(1. 2.)를 쓴다.
- 일반 문장과 필요 시 하이픈(-) 목록만 사용한다.
- 영문 컬럼명·코드명(temp_avg 등)은 사용자에게 쓰지 말고, JSON에 적힌 한글 이름만 쓴다.
- 반드시 가격_예측_요약의 선택한_작물, 선택한_단위명, 선택한_등급명을 기준으로 분석한다.

내용 우선순위(이 순서로 작성):
1) 10일 예측값만으로 증감 방향 판단
   - 가격_예측_요약 안의 앞으로_예측된_가격_개요·예측_전체_가격_방향·변동_퍼센트 등을 인용한다.
   - 예측이 없으면 예측 구간 없음이라고 쓴다.

2) 제공 자료로 증감 원인 추정
   - 기상_경제_뉴스_자료 안의 기상과_경제_지표_요약과 직전_같은_길이_기간과_비교한_평균_변화를 근거로, 가격 방향과 맞는지 추정한다.
   - 수치_읽는_법을 따른다. 절대값이 아니라 최근 vs 이전의 큰지/작은지 같은 상대 비교만 한다.
   - 참고_뉴스_글이 있을 때만 뉴스 내용을 인용한다. 뉴스_안내가 있으면 뉴스는 쓰지 않는다.

3) 한 줄 한계: 모델·데이터 한계와 불확실성을 짧게 명시한다.

4) XAI용 핵심 근거 3문장:
   - 가격 방향 근거: 10일 예측의 첫값·마지막값·평균 또는 변동 퍼센트로 왜 상승/하락/횡보인지 설명한다.
   - 변동성 근거: 최근 실제 가격 범위, 예측 범위, 변동성·리스크가 왜 큰지/작은지 설명한다.
   - 외부요인 근거: 기상·경제 지표 중 가격 방향 또는 변동성과 연결되는 요인을 설명한다.
   - 모델명과 모델 구조 설명은 쓰지 말고, TimesFM은 점예측, Chronos2는 구간 예측이라고만 쓴다.

금지: 투자 조언, 확정적 예언, JSON에 없는 숫자 만들기."""

    xai_prompt_builder = _load_xai_prompt_builder()
    use_xai_prompt = False
    system_prompt = fallback_system
    user_prompt = "아래 JSON만 근거로 위 규칙에 맞게 작성하라.\n\n" + repr(user_payload)
    if xai_prompt_builder:
        try:
            item_forecasts = _build_item_forecasts_for_xai(item_id, chart_summary)
            context_summary = summarize_baseline_context_for_explain(item_id)
            warnings_block = _build_xai_warnings_block(item_id, cov)
            report_excerpt = _build_xai_report_excerpt(item_id, cov)
            system_prompt = xai_prompt_builder.build_system_prompt()
            user_prompt = xai_prompt_builder.build_user_prompt(
                item_id=item_id,
                item_forecasts=item_forecasts,
                context_summary=context_summary,
                warnings_block=warnings_block,
                report_excerpt=report_excerpt,
            )
            user_prompt += (
                "\n\n[대시보드 XAI 표시 지침]\n"
                "forecast_explanation은 수치 나열·표 요약이 아니라 **판단 근거** 중심으로 작성한다.\n"
                "첫 문단(3일): 왜 모델이 해당 점예측 수준·경로를 택했는지, 왜 상승/하락/횡보로 해석하는지 "
                "데이터·변수·외부자료 기반 인과 추론으로 설명한다.\n"
                "둘째 문단(10일): 왜 예측 범위가 넓거나 좁은지, 중심값이 왜 그 방향인지, "
                "변동성·리스크 판단 근거와 기상·경제·수급 요인 연결을 설명한다.\n"
                "위 user prompt에 기상·경제 지표·365일 context가 제공되면 "
                "'기상·경제 자료 없음', '뉴스·월보 연결 없음'처럼 단정하지 말고 "
                "제공된 지표를 근거로 설명한다.\n"
                "금지: 날짜·가격·분위수만 줄줄이 읊는 수치해석, forecast_summary 숫자 반복, "
                "모델명·구조·파라미터 설명.\n"
                "수치는 인과 설명에 꼭 필요할 때만 최소한으로 인용한다.\n"
                "JSON에 아래 필드를 반드시 추가 포함:\n"
                "- forecast_summary_plain: forecast_summary와 같은 내용, 농부·시장 상인도 이해하는 쉬운 말\n"
                "- forecast_explanation_plain: forecast_explanation과 같은 내용, 쉬운 말\n"
                "쉬운 말 규칙: 어려운 전문용어·영어·모델명 금지, 짧은 문장, 존댓말, "
                "내용(판단·방향·요인)은 전문가용과 동일하게 유지.\n"
            )
            use_xai_prompt = True
        except Exception as exc:
            print(f"[agri_gpt] xai_explainer prompt build failed; using fallback prompt: {exc.__class__.__name__}: {exc}")
            use_xai_prompt = False
            system_prompt = fallback_system
            user_prompt = "아래 JSON만 근거로 위 규칙에 맞게 작성하라.\n\n" + repr(user_payload)

    try:
        if use_xai_prompt:
            m = (model or "").lower()
            uses_completion_tokens = (
                m.startswith("gpt-5")
                or m.startswith("o1")
                or m.startswith("o3")
                or "gpt-5" in m
            )
            token_kw = (
                {"max_completion_tokens": 2800}
                if uses_completion_tokens
                else {"max_tokens": 2800}
            )
            try:
                r = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.35,
                    **token_kw,
                )
            except OpenAIError:
                r = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.35,
                    max_tokens=2800,
                )
            raw = (r.choices[0].message.content or "").strip()
            parsed = _parse_response_json(raw)
            if parsed and isinstance(parsed, dict):
                expert, plain, explanation_plain = _compose_explanation_from_parsed(parsed)
                if expert:
                    if not explanation_plain:
                        plain = _simplify_explanation_for_laypeople(client, model, expert)
                    return _explain_result(expert, plain)
            expert = _sanitize_explanation_output(raw)
            return _explain_result(expert, _simplify_explanation_for_laypeople(client, model, expert))

        try:
            r = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.35,
                max_tokens=2800,
            )
        except OpenAIError:
            r = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.35,
                max_tokens=2800,
            )
    except AuthenticationError:
        msg = (
            "GPT 설명을 생성하지 못했습니다.\n"
            "서버의 OPENAI_API_KEY가 유효하지 않습니다. .env의 키를 새 OpenAI API 키로 교체한 뒤 서버를 재시작해주세요."
        )
        return _explain_result(msg)
    except OpenAIError as exc:
        msg = f"GPT 설명을 생성하지 못했습니다. OpenAI API 오류: {exc.__class__.__name__}"
        return _explain_result(msg)
    raw = (r.choices[0].message.content or "").strip()
    expert = _sanitize_explanation_output(raw)
    return _explain_result(expert, _simplify_explanation_for_laypeople(client, model, expert))


def generate_analysis_image(item_id: str, explanation: str) -> dict[str, str] | None:
    """Generate a compact visual card for the analysis when the image API is available."""
    fallback = _fallback_analysis_image(item_id, explanation)
    api_key = getattr(settings, "OPENAI_API_KEY", "") or ""
    use_openai_image = bool(getattr(settings, "AGRI_USE_OPENAI_IMAGE", False))
    if not api_key.strip() or not use_openai_image:
        return fallback
    from openai import OpenAI, OpenAIError

    model = getattr(settings, "OPENAI_IMAGE_MODEL", "gpt-image-2")
    try:
        meta = item_option_meta(item_id)
        item_label = meta.get("label") or item_id
    except Exception:
        item_label = item_id
    prompt = (
        "Create a polished Korean agricultural price forecast infographic card. "
        "Style: dark architecture-diagram aesthetic, clean SVG-like blocks, no clutter. "
        "Include: overall summary, price direction, volatility/risk, weather/economic factors. "
        "Use minimal Korean text, readable labels, green/amber/cyan palette. "
        f"Selected item: {item_label}. Context: {str(explanation or '')[:1200]}"
    )
    try:
        client = OpenAI(api_key=api_key, timeout=12)
        result = client.images.generate(
            model=model,
            prompt=prompt,
            size="1024x1024",
        )
    except OpenAIError:
        return fallback
    except Exception:
        return fallback

    data = (getattr(result, "data", None) or [None])[0]
    if data is None:
        return fallback
    b64 = getattr(data, "b64_json", None)
    if b64:
        return {"data_url": f"data:image/png;base64,{b64}", "model": model}
    url = getattr(data, "url", None)
    if url:
        return {"url": url, "model": model}
    return fallback


def _svg_text(value: Any, limit: int = 34) -> str:
    text = str(value or "-").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return text[:limit] + ("..." if len(text) > limit else "")


def _svg_lines(value: Any, limit: int = 24, max_lines: int = 3, ellipsis: bool = True) -> list[str]:
    text = re.sub(r"\s+", " ", str(value or "-")).strip()
    words = text.split(" ")
    lines: list[str] = []
    current = ""
    for word in words:
        if len(word) > limit:
            if current:
                lines.append(current)
                current = ""
            for i in range(0, len(word), limit):
                lines.append(word[i : i + limit])
                if len(lines) >= max_lines:
                    break
            if len(lines) >= max_lines:
                break
            continue
        candidate = f"{current} {word}".strip()
        if len(candidate) <= limit:
            current = candidate
            continue
        if current:
            lines.append(current)
        current = word[:limit]
        if len(lines) >= max_lines:
            break
    if current and len(lines) < max_lines:
        lines.append(current)
    if not lines:
        lines = ["-"]
    if ellipsis and len(lines) == max_lines and len(" ".join(words)) > len(" ".join(lines)):
        lines[-1] = lines[-1][: max(0, limit - 3)] + "..."
    return [_svg_text(line, limit + 3) for line in lines]


def _context_payload(context: str) -> dict[str, Any]:
    try:
        payload = json.loads(context or "{}")
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    return {"analysis_text": context or ""}


def _pick_sentence(text: str, keywords: list[str], default: str) -> str:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    parts = [p.strip(" -") for p in re.split(r"(?<=[.!?。])\s+|[。\n]", cleaned) if p.strip()]
    for keyword in keywords:
        for part in parts:
            if keyword in part:
                return part
    return parts[0] if parts else default


def _money(value: Any) -> str:
    try:
        return f"{float(value):,.0f}원"
    except Exception:
        return "-"


def _risk_label(value: Any) -> str:
    try:
        pct = float(value)
    except Exception:
        return "자료 부족"
    if pct >= 12:
        return "높음"
    if pct >= 6:
        return "보통"
    return "낮음"


def _factor_summary_lines(text: str, kind: str) -> list[str]:
    raw = str(text or "")
    if kind == "weather":
        factors: list[str] = []
        if re.search(r"강수|비|우천|장마|눈", raw):
            factors.append("강수")
        if re.search(r"기온|온도|한파|폭염", raw):
            factors.append("기온")
        if re.search(r"습도", raw):
            factors.append("습도")
        if re.search(r"일조|햇볕", raw):
            factors.append("일조")
        head = "·".join(factors[:2]) or "기상"
        return [
            f"{head} 변화 확인",
            "출하량·품질 변동 가능",
        ]
    factors = []
    if re.search(r"수요|소비", raw):
        factors.append("수요")
    if re.search(r"공급|출하|출하량|재고", raw):
        factors.append("공급")
    if re.search(r"물류|유가|운송", raw):
        factors.append("물류비")
    if re.search(r"환율|물가|금리", raw):
        factors.append("거시")
    head = "·".join(factors[:2]) or "시장"
    return [
        f"{head} 흐름 확인",
        "가격·거래량 영향 가능",
    ]


def _text_block(lines: list[str], x: int, y: int, fill: str, size: int = 24, weight: int = 800) -> str:
    return "\n".join(
        f'<text x="{x}" y="{y + i * (size + 12)}" fill="{fill}" font-size="{size}" font-family="Arial, sans-serif" font-weight="{weight}">{line}</text>'
        for i, line in enumerate(lines)
    )


def _fallback_analysis_image(item_id: str, context: str) -> dict[str, str]:
    try:
        meta = item_option_meta(item_id)
    except Exception:
        meta = {"label": item_id, "crop": item_id, "unit": "-", "grade": "-"}
    payload = _context_payload(context)
    crop = payload.get("crop") or meta.get("crop") or item_id
    unit = payload.get("unit") or meta.get("unit") or "-"
    grade = payload.get("grade") or meta.get("grade") or "-"
    analysis_text = str(payload.get("analysis_text") or "")
    title_lines = _svg_lines(f"{crop} · {unit} · {grade} 가격 전망", 17, 2, ellipsis=False)
    direction = _svg_text(payload.get("direction") or "예측값 부족", 8)
    risk = _risk_label(payload.get("volatility_pct"))
    current_price = _money(payload.get("current_actual_krw"))
    avg_price = _money(payload.get("prediction_average_krw"))
    summary_text = f"현재가 {current_price} 기준, 예측 평균은 {avg_price}입니다."
    outlook_text = f"{direction} 흐름을 중심으로 단기 가격을 확인하세요."
    risk_text = f"변동성 {payload.get('volatility_pct') if payload.get('volatility_pct') is not None else '-'}%"
    weather_text = _pick_sentence(
        analysis_text,
        ["날씨", "기온", "강수", "습도", "일조"],
        "기상 조건 변화가 출하와 품질에 영향을 줄 수 있습니다.",
    )
    economy_text = _pick_sentence(
        analysis_text,
        ["경제", "유가", "환율", "물류", "수요", "공급"],
        "수요, 물류비, 시장 공급량 변화를 함께 확인해야 합니다.",
    )
    weather_lines = [_svg_text(line, 18) for line in _factor_summary_lines(weather_text, "weather")]
    economy_lines = [_svg_text(line, 18) for line in _factor_summary_lines(economy_text, "economy")]
    origin_date = _svg_text(payload.get("origin_date") or "-", 16)
    summary_lines = _svg_lines(summary_text, 12, 3)
    outlook_lines = _svg_lines(outlook_text, 13, 2)
    risk_lines = _svg_lines(f"{risk_text} · {risk}", 12, 2)
    try:
        avg = float(payload.get("prediction_average_krw") or 0)
        vol = float(payload.get("volatility_pct") or 0)
        low = avg * max(0.0, 1 - vol / 100)
        high = avg * (1 + vol / 100)
        forecast_range = f"{low:,.0f} ~ {high:,.0f}원"
    except Exception:
        forecast_range = "-"
    direction_color = "#bef264" if "상" in direction else "#60a5fa" if "하" in direction else "#facc15"
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="1024" height="1024" viewBox="0 0 1024 1024">
<defs>
<linearGradient id="bg" x1="0" x2="1" y1="0" y2="1"><stop offset="0" stop-color="#04111f"/><stop offset=".55" stop-color="#07130f"/><stop offset="1" stop-color="#0b1220"/></linearGradient>
<filter id="shadow"><feDropShadow dx="0" dy="16" stdDeviation="14" flood-color="#000" flood-opacity=".34"/></filter>
<clipPath id="rangeCardClip"><rect x="687" y="245" width="248" height="218" rx="12"/></clipPath>
</defs>
<rect width="1024" height="1024" fill="url(#bg)"/>
<rect x="28" y="24" width="968" height="964" rx="20" fill="none" stroke="#496168" stroke-width="2"/>
<g opacity=".08" stroke="#dfffd0"><path d="M48 152H976M48 296H976M48 440H976M48 584H976M48 728H976M48 872H976M168 48V962M328 48V962M488 48V962M648 48V962M808 48V962"/></g>

<circle cx="90" cy="86" r="40" fill="#17391d" stroke="#bef264" stroke-width="2"/>
<path d="M69 86C86 52 120 58 123 90C101 78 84 81 69 86Z" fill="#a3e635"/>
<path d="M70 90C88 92 108 101 122 116C96 120 75 110 70 90Z" fill="#84cc16"/>
{_text_block(title_lines, 144, 68, "#f8fafc", 24, 900)}
<text x="144" y="134" fill="#cbd5e1" font-size="17" font-family="Arial, sans-serif">선택 품목 기준 AI 가격 예측 인포그래픽</text>
<rect x="735" y="44" width="218" height="94" rx="10" fill="#101827" stroke="#475569"/>
<text x="758" y="80" fill="#cbd5e1" font-size="16" font-family="Arial" font-weight="800">예측 기준일</text>
<text x="932" y="80" fill="#ffffff" font-size="16" font-family="Arial" font-weight="900" text-anchor="end">{origin_date}</text>
<text x="758" y="116" fill="#cbd5e1" font-size="16" font-family="Arial" font-weight="800">통화</text>
<text x="932" y="116" fill="#ffffff" font-size="16" font-family="Arial" font-weight="900" text-anchor="end">KRW (원)</text>

<g filter="url(#shadow)">
<rect x="54" y="164" width="306" height="360" rx="12" fill="#082620" stroke="#84cc16" stroke-width="2"/>
<circle cx="88" cy="202" r="18" fill="#bef264"/><text x="82" y="209" fill="#082620" font-size="21" font-family="Arial" font-weight="900">1</text>
<text x="118" y="210" fill="#bef264" font-size="23" font-family="Arial" font-weight="900">전체 요약</text>
{_text_block(summary_lines, 82, 266, "#f8fafc", 17, 800)}
<text x="92" y="388" fill="#bef264" font-size="18" font-family="Arial" font-weight="900">✓ 현재가</text>
<text x="180" y="388" fill="#bef264" font-size="22" font-family="Arial" font-weight="900">{current_price}</text>
<text x="92" y="448" fill="#bef264" font-size="18" font-family="Arial" font-weight="900">✓ 예측 평균</text>
<text x="208" y="448" fill="#bef264" font-size="22" font-family="Arial" font-weight="900">{avg_price}</text>

<rect x="378" y="164" width="592" height="360" rx="12" fill="#081523" stroke="#38bdf8" stroke-width="2"/>
<circle cx="412" cy="202" r="18" fill="#bef264"/><text x="406" y="209" fill="#082620" font-size="21" font-family="Arial" font-weight="900">2</text>
<text x="442" y="210" fill="#bef264" font-size="24" font-family="Arial" font-weight="900">가격 전망</text>
<rect x="410" y="244" width="250" height="220" rx="12" fill="#092a22" stroke="#84cc16"/>
<text x="446" y="290" fill="#bef264" font-size="20" font-family="Arial" font-weight="900">향후 3일 점예측</text>
<text x="442" y="340" fill="#bef264" font-size="29" font-family="Arial" font-weight="900">{avg_price}</text>
<polyline points="438,414 488,382 540,392 608,356" fill="none" stroke="#bef264" stroke-width="8" stroke-linecap="round" stroke-linejoin="round"/>
<circle cx="438" cy="414" r="6" fill="#ffffff"/><circle cx="488" cy="382" r="6" fill="#ffffff"/><circle cx="540" cy="392" r="6" fill="#ffffff"/><circle cx="608" cy="356" r="6" fill="#ffffff"/>
<rect x="686" y="244" width="250" height="220" rx="12" fill="#0b1f33" stroke="#38bdf8"/>
<text x="720" y="290" fill="#67e8f9" font-size="20" font-family="Arial" font-weight="900">향후 10일 범위예측</text>
<text x="714" y="340" fill="#f8fafc" font-size="22" font-family="Arial" font-weight="900">{forecast_range}</text>
<g clip-path="url(#rangeCardClip)">
<path d="M716 414C770 374 832 398 910 352L910 432C838 448 770 440 716 456Z" fill="#38bdf8" opacity=".42"/>
<polyline points="716,436 760,414 812,406 862,382 910,364" fill="none" stroke="#f8fafc" stroke-width="5" stroke-dasharray="8 8"/>
</g>
</g>

<g filter="url(#shadow)">
<rect x="54" y="544" width="306" height="164" rx="12" fill="#0b2b1c" stroke="#84cc16" stroke-width="2"/>
<circle cx="88" cy="582" r="18" fill="#bef264"/><text x="82" y="589" fill="#082620" font-size="21" font-family="Arial" font-weight="900">3</text>
<text x="118" y="590" fill="#bef264" font-size="25" font-family="Arial" font-weight="900">가격 방향</text>
<circle cx="136" cy="650" r="40" fill="none" stroke="#bef264" stroke-width="4"/>
<path d="M108 662L128 642L145 656L164 632" fill="none" stroke="{direction_color}" stroke-width="9" stroke-linecap="round" stroke-linejoin="round"/>
<path d="M150 632H164V646" fill="none" stroke="{direction_color}" stroke-width="9" stroke-linecap="round" stroke-linejoin="round"/>
<text x="198" y="642" fill="{direction_color}" font-size="29" font-family="Arial" font-weight="900">{direction}</text>
{_text_block(outlook_lines, 198, 674, "#d9f99d", 14, 800)}

<rect x="378" y="544" width="592" height="164" rx="12" fill="#1f1b0b" stroke="#f59e0b" stroke-width="2"/>
<circle cx="412" cy="582" r="18" fill="#fde047"/><text x="406" y="589" fill="#082620" font-size="21" font-family="Arial" font-weight="900">4</text>
<text x="442" y="590" fill="#fde047" font-size="25" font-family="Arial" font-weight="900">변동성 / 리스크</text>
<path d="M430 680A58 58 0 0 1 546 680" fill="none" stroke="#475569" stroke-width="18"/>
<path d="M430 680A58 58 0 0 1 526 638" fill="none" stroke="#fde047" stroke-width="18"/>
<line x1="488" y1="680" x2="528" y2="644" stroke="#facc15" stroke-width="6" stroke-linecap="round"/>
<circle cx="488" cy="680" r="9" fill="#facc15"/>
<text x="592" y="638" fill="#f8fafc" font-size="23" font-family="Arial" font-weight="900">리스크 {risk}</text>
{_text_block(risk_lines, 592, 672, "#fde68a", 16, 800)}
</g>

<g filter="url(#shadow)">
<rect x="54" y="730" width="916" height="172" rx="12" fill="#071d2a" stroke="#38bdf8" stroke-width="2"/>
<circle cx="88" cy="768" r="18" fill="#67e8f9"/><text x="82" y="775" fill="#082620" font-size="21" font-family="Arial" font-weight="900">5</text>
<text x="118" y="776" fill="#67e8f9" font-size="25" font-family="Arial" font-weight="900">주요 요인</text>
<rect x="94" y="792" width="405" height="94" rx="10" fill="#0b2737" stroke="#38bdf8"/>
<text x="124" y="820" fill="#dffafe" font-size="18" font-family="Arial" font-weight="900">날씨적 요인</text>
{_text_block(weather_lines, 124, 848, "#f8fafc", 13, 800)}
<rect x="526" y="792" width="405" height="94" rx="10" fill="#0b2737" stroke="#38bdf8"/>
<text x="556" y="820" fill="#dffafe" font-size="18" font-family="Arial" font-weight="900">경제적 요인</text>
{_text_block(economy_lines, 556, 848, "#f8fafc", 13, 800)}
</g>

<rect x="54" y="912" width="916" height="48" rx="10" fill="#101827" stroke="#475569"/>
<text x="86" y="943" fill="#cbd5e1" font-size="17" font-family="Arial" font-weight="800">본 전망은 저장 데이터 및 AI 예측 모델 기반의 참고 정보입니다. 실제 거래가는 시장 상황에 따라 달라질 수 있습니다.</text>
</svg>"""
    return {
        "data_url": "data:image/svg+xml;charset=utf-8," + quote(svg),
        "model": "fallback-svg",
    }
