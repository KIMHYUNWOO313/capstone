"""농업 월보 PDF 텍스트 추출 및 품목별 관련 문단 인덱싱."""
import re
from pathlib import Path

from pypdf import PdfReader

from config import AGRI_REPORT_DIR

CROP_KEYWORDS_KR: dict[str, list[str]] = {
    "apple":            ["사과", "후지"],
    "cabbage":          ["양배추"],
    "carrot":           ["당근"],
    "crown_daisy":      ["쑥갓"],
    "cucumber_bdadagi": ["오이", "다다기"],
    "garlic_chive":     ["부추"],
    "honewort":         ["미나리"],
    "napa_cabbage":     ["배추"],
    "onion":            ["양파"],
    "perilla_leaf":     ["깻잎"],
    "potato_sumi":      ["감자", "수미"],
    "spinach":          ["시금치"],
    "sweetpotato":      ["고구마"],
}


def _extract_pdf_text(path: Path) -> str:
    try:
        reader = PdfReader(str(path))
        chunks = []
        for page in reader.pages:
            try:
                chunks.append(page.extract_text() or "")
            except Exception:
                continue
        return "\n".join(chunks)
    except Exception as e:
        return f"(PDF 추출 실패: {path.name} — {e})"


def load_all_reports() -> dict[str, str]:
    """Returns: {pdf_filename: full_text}"""
    out = {}
    if not AGRI_REPORT_DIR.exists():
        return out
    for p in sorted(AGRI_REPORT_DIR.glob("*.pdf")):
        out[p.name] = _extract_pdf_text(p)
    return out


def extract_relevant_paragraphs(
    text: str,
    keywords: list[str],
    window: int = 1,
    max_paragraphs: int = 6,
) -> list[str]:
    """키워드 포함 문단(±window) 추출. 중복 제거 후 max_paragraphs로 컷."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    hits: set[int] = set()
    for i, para in enumerate(paragraphs):
        if any(kw in para for kw in keywords):
            for j in range(max(0, i - window), min(len(paragraphs), i + window + 1)):
                hits.add(j)
    indices = sorted(hits)[:max_paragraphs]
    return [paragraphs[i] for i in indices]


def relevant_excerpts_for_crop(
    reports: dict[str, str], crop: str, max_chars: int = 2000
) -> str:
    """단일 작물에 대한 모든 PDF의 관련 발췌를 합쳐 반환."""
    keywords = CROP_KEYWORDS_KR.get(crop, [crop])
    chunks = []
    for fname, text in reports.items():
        paras = extract_relevant_paragraphs(text, keywords)
        if not paras:
            continue
        chunks.append(f"### {fname}\n" + "\n\n".join(paras))
    joined = "\n\n".join(chunks).strip()
    if not joined:
        return "(관련 월보 발췌 없음)"
    if len(joined) > max_chars:
        joined = joined[:max_chars] + " …(생략)"
    return joined


if __name__ == "__main__":
    reps = load_all_reports()
    print(f"PDF: {list(reps.keys())}")
    print("\n=== apple 발췌 ===")
    print(relevant_excerpts_for_crop(reps, "apple"))
