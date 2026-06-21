# -*- coding: utf-8 -*-
"""재정경제부(구 기획재정부) 일일경제지표 게시판 크롤러 + HWP 파싱.

데이터 흐름:
    목록 페이지(pageIndex=1..N)
        → 각 게시물 nttId, 제목, 등록일 추출
        → 제목에서 데이터 일자 파싱 (예: "일일경제지표('26.4.30)" → 2026-04-30)
    상세 페이지(detailTbEconomyIndicatorView.do?searchNttId1=<nttId>)
        → 첨부파일 atchFileId, fileSn, 원본 파일명 추출
    파일 다운로드(FileDown.do?atchFileId=...&fileSn=...)
        → data/mofe_daily/hwp/<YYYYMMDD>.hwp 로 저장
    HWP → HTML 변환(pyhwp.hwp5html)
        → data/mofe_daily/html/<YYYYMMDD>/index.xhtml
    HTML 표 → CSV (3종)
        ① 보고서 형태 wide:  data/mofe_daily/csv/wide/<YYYYMMDD>.csv
              한 게시물의 6개 표(금리/주가/환율/국제금리·주가·가산금리/유가·곡물·원자재/반도체)를
              [섹션] / 헤더행 / 항목행 들 로 위→아래 그대로 복원. 셀 병합(rowspan/colspan)도 펼침.
        ② long format:       data/mofe_daily/csv/<YYYYMMDD>.csv
              (data_date, source_url, table_idx, table_section, row, col, header_top, header_left, value)
              분석/조인용. table_section 은 표 직전 단락에서 추출한 섹션명.
        ③ 섹션별 시계열 wide: data/mofe_daily/wide/<섹션>.csv
              행=항목(콜금리, KOSPI, ₩/U$ ...), 열=일자(2016-01-04 ... 2026-04-30).
              매 보고서에서 그 일자에 해당하는 컬럼(예: '4/30')을 자동으로 골라 채운다.

    data/mofe_daily/index.csv  — 메타 누적 (data_date, ntt_id, title, atch_file_id, hwp_path, ...)
"""
from __future__ import annotations

import csv
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


BASE_URL = "https://mofe.go.kr"
LIST_URL = f"{BASE_URL}/st/ecnmyidx/TbEconomyIndicatorList.do"
DETAIL_URL = f"{BASE_URL}/st/ecnmyidx/detailTbEconomyIndicatorView.do"
FILE_DOWN_URL = f"{BASE_URL}/com/cmm/fms/FileDown.do"
BBS_ID = "MOSFBBS_000000000045"
MENU_NO = "6010200"

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124 Safari/537.36",
    "Accept-Language": "ko-KR,ko;q=0.9",
}

DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "mofe_daily"
HWP_DIR = DATA_ROOT / "hwp"
HTML_DIR = DATA_ROOT / "html"
CSV_DIR = DATA_ROOT / "csv"
WIDE_DIR = CSV_DIR / "wide"  # 일자별 wide(보고서 형태) CSV
SERIES_DIR = DATA_ROOT / "wide"  # 섹션별 시계열 wide CSV
INDEX_CSV = DATA_ROOT / "index.csv"


# ---- 데이터 모델 -----------------------------------------------------------


@dataclass
class ListItem:
    """목록 페이지에서 뽑은 게시물 한 건."""

    ntt_id: str
    title: str
    registered_at: str  # 'YYYY.MM.DD.' 원문
    data_date: date | None  # 제목에서 파싱한 데이터 일자


@dataclass
class Attachment:
    atch_file_id: str
    file_sn: str
    file_name: str  # 한글 원본 파일명


# ---- 목록 / 상세 / 첨부 ---------------------------------------------------


_TITLE_DATE_PATTERNS = [
    re.compile(r"['‘’](\d{2})\.\s*(\d{1,2})\.\s*(\d{1,2})\)?"),
    re.compile(r"\((\d{4})\.\s*(\d{1,2})\.\s*(\d{1,2})\.?\)"),
    re.compile(r"(\d{4})\.\s*(\d{1,2})\.\s*(\d{1,2})\.?"),
]


def _parse_title_date(title: str) -> date | None:
    """제목 문자열에서 데이터 일자를 추출.

    예: "일일경제지표('26.4.30)"      → 2026-04-30
        "일일경제지표(2024.10. 7)"    → 2024-10-07
        "일일경제지표(2015. 1. 5.)"   → 2015-01-05
    """

    for rgx in _TITLE_DATE_PATTERNS:
        m = rgx.search(title)
        if not m:
            continue
        a, b, c = m.groups()
        try:
            year = int(a)
            month = int(b)
            day = int(c)
        except ValueError:
            continue
        if year < 100:
            year = 2000 + year
        try:
            return date(year, month, day)
        except ValueError:
            continue
    return None


_LIST_ITEM_RGX = re.compile(
    r"fn_egov_select\(['\"](MOSF_\d+)['\"]\)[^>]*>([^<]+)</a>"
    r"(?:.|\n){0,400}?(\d{4}\.\s*\d{1,2}\.\s*\d{1,2}\.)",
    re.MULTILINE,
)


def fetch_list_page(session: requests.Session, page_index: int) -> list[ListItem]:
    """단일 목록 페이지에서 게시물 목록 추출."""

    r = session.get(
        LIST_URL,
        params={
            "bbsId": BBS_ID,
            "menuNo": MENU_NO,
            "pageIndex": str(page_index),
        },
        headers=DEFAULT_HEADERS,
        timeout=30,
    )
    r.raise_for_status()
    items: list[ListItem] = []
    for m in _LIST_ITEM_RGX.finditer(r.text):
        ntt_id = m.group(1).strip()
        title = m.group(2).strip()
        registered_at = m.group(3).strip()
        data_date = _parse_title_date(title)
        items.append(
            ListItem(
                ntt_id=ntt_id,
                title=title,
                registered_at=registered_at,
                data_date=data_date,
            )
        )
    return items


_ATTACH_RGX = re.compile(
    r"FileDown\.do[^?]*\?atchFileId=(ATCH_\d+)&(?:amp;)?fileSn=(\d+)",
    re.IGNORECASE,
)


def fetch_detail_attachments(session: requests.Session, ntt_id: str) -> list[Attachment]:
    r = session.get(
        DETAIL_URL,
        params={
            "bbsId": BBS_ID,
            "searchNttId1": ntt_id,
            "menuNo": MENU_NO,
        },
        headers=DEFAULT_HEADERS,
        timeout=30,
    )
    r.raise_for_status()
    html = r.text
    seen: set[tuple[str, str]] = set()
    out: list[Attachment] = []
    soup = BeautifulSoup(html, "html.parser")
    # 첨부파일 영역 우선 탐색
    for a in soup.select("div.fileInfo a[href*='FileDown.do']"):
        href = a.get("href", "")
        m = _ATTACH_RGX.search(href)
        if not m:
            continue
        key = (m.group(1), m.group(2))
        if key in seen:
            continue
        seen.add(key)
        name = (a.get_text(strip=True) or "").strip()
        out.append(Attachment(atch_file_id=key[0], file_sn=key[1], file_name=name))
    if not out:
        for m in _ATTACH_RGX.finditer(html):
            key = (m.group(1), m.group(2))
            if key in seen:
                continue
            seen.add(key)
            out.append(Attachment(atch_file_id=key[0], file_sn=key[1], file_name=""))
    return out


def download_attachment(
    session: requests.Session, att: Attachment, out_path: Path
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with session.get(
        FILE_DOWN_URL,
        params={"atchFileId": att.atch_file_id, "fileSn": att.file_sn},
        headers=DEFAULT_HEADERS,
        timeout=60,
        stream=True,
    ) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)
    return out_path


# ---- HWP → HTML 변환 ------------------------------------------------------


_HWP_RUNNER = r"""
import logging, sys
for n in (
    'hwp5','hwp5.binmodel','hwp5.bintype','hwp5.dataio',
    'hwp5.recordstream','hwp5.xmlmodel','hwp5.hwp5html',
    'hwp5.tagids','hwp5.compressed','hwp5.utils',
):
    lg = logging.getLogger(n); lg.setLevel(logging.CRITICAL); lg.propagate = False
import warnings
warnings.simplefilter('ignore')
from hwp5.hwp5html import main
sys.argv = ['hwp5html', '--output', sys.argv[1], sys.argv[2]]
try:
    main()
except SystemExit:
    pass
"""


def hwp_to_html(hwp_path: Path, out_dir: Path, *, timeout: float = 60.0) -> Path:
    """pyhwp의 hwp5html.main을 자식 파이썬 프로세스에서 호출해 index.xhtml을 만든다.

    오래된/손상된 HWP 에서 pyhwp 가 동일 경고를 무한 출력해 멈추는 사례가 있어
    자식 프로세스로 띄워 ``timeout`` 초가 지나면 강제 종료한다.
    실패 시 ``RuntimeError`` 를 던진다.
    """

    import subprocess
    import sys

    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-c",
        _HWP_RUNNER,
        str(out_dir),
        str(hwp_path),
    ]
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"hwp5html timeout({timeout}s) for {hwp_path}") from exc

    out = out_dir / "index.xhtml"
    if proc.returncode != 0 or not out.exists():
        raise RuntimeError(
            f"hwp5html failed (rc={proc.returncode}) for {hwp_path}"
        )
    return out


# ---- HTML 표 → CSV --------------------------------------------------------


_NORM_WS = re.compile(r"\s+")


def _norm_text(s: str) -> str:
    return _NORM_WS.sub(" ", (s or "").strip())


def parse_tables_to_rows(
    html_path: Path,
    *,
    data_date: date,
    source_url: str,
) -> list[dict]:
    """xhtml에서 모든 표를 long-format CSV row로 풀어낸다."""

    soup = BeautifulSoup(html_path.read_text(encoding="utf-8"), "html.parser")
    rows: list[dict] = []

    tables = soup.find_all("table")

    def _section_for(table) -> str:
        # 표 앞쪽에 위치한 p 단락을 문서 순서대로 거꾸로 훑어 [..] 형태를 찾는다.
        for prev in table.find_all_previous("p"):
            txt = _norm_text(prev.get_text(" "))
            m = re.search(r"\[([^\]]+)\]", txt)
            if m:
                return m.group(1).strip()
        return ""

    for ti, table in enumerate(tables):
        section = _section_for(table)
        trs = table.find_all("tr")
        if not trs:
            continue
        # 헤더(top): 첫 행, 헤더(left): 첫 열
        first_row_cells = [
            _norm_text(td.get_text(" ")) for td in trs[0].find_all(["td", "th"])
        ]
        for ri, tr in enumerate(trs):
            tds = tr.find_all(["td", "th"])
            for ci, td in enumerate(tds):
                value = _norm_text(td.get_text(" "))
                header_top = (
                    first_row_cells[ci] if ri > 0 and ci < len(first_row_cells) else ""
                )
                header_left = ""
                if ci > 0 and tds:
                    header_left = _norm_text(tds[0].get_text(" "))
                rows.append(
                    {
                        "data_date": data_date.isoformat(),
                        "source_url": source_url,
                        "table_idx": ti,
                        "table_section": section,
                        "row": ri,
                        "col": ci,
                        "header_top": header_top,
                        "header_left": header_left,
                        "value": value,
                    }
                )
    return rows


def write_csv(rows: list[dict], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "data_date",
        "source_url",
        "table_idx",
        "table_section",
        "row",
        "col",
        "header_top",
        "header_left",
        "value",
    ]
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return out_path


# ---- Wide CSV (보고서 형태 그대로) ---------------------------------------


_SECTION_NORMALIZE_MAP = {
    # 표기 변형 → 정규화된 라벨
    "국제금리주가가산금리": "국제금리·주가·가산금리",
    "유가곡물원자재": "유가·곡물·원자재",
}


def _normalize_section_label(label: str) -> str:
    if not label:
        return ""
    # 가운뎃점/점·공백 제거 후 매칭
    key = re.sub(r"[·ㆍ․\.\s]+", "", label)
    return _SECTION_NORMALIZE_MAP.get(key, label.strip())


def parse_tables_to_grids(html_path: Path) -> list[tuple[str, list[list[str]]]]:
    """xhtml의 모든 표를 (section_label, 2D grid) 튜플 리스트로 반환.

    rowspan/colspan 은 그대로 펼쳐 빈 셀이 없도록 채운다(셀 값 복제).
    section_label 은 표 직전의 [..] 단락에서 추출.
    """

    soup = BeautifulSoup(html_path.read_text(encoding="utf-8"), "html.parser")
    out: list[tuple[str, list[list[str]]]] = []

    def _section_for(table) -> str:
        for prev in table.find_all_previous("p"):
            txt = _norm_text(prev.get_text(" "))
            m = re.search(r"\[([^\]]+)\]", txt)
            if m:
                return m.group(1).strip()
        return ""

    for table in soup.find_all("table"):
        trs = table.find_all("tr")
        if not trs:
            continue
        # rowspan/colspan 펼치기 — placement[r][c] = text
        # 사전 패스로 최대 컬럼 수 추정
        max_cols = 0
        for tr in trs:
            count = 0
            for td in tr.find_all(["td", "th"]):
                count += int(td.get("colspan", 1) or 1)
            max_cols = max(max_cols, count)

        rows = len(trs)
        grid: list[list[str | None]] = [[None] * max_cols for _ in range(rows)]
        for ri, tr in enumerate(trs):
            ci = 0
            for td in tr.find_all(["td", "th"]):
                # 이미 채워진 칸은 건너뛴다
                while ci < max_cols and grid[ri][ci] is not None:
                    ci += 1
                if ci >= max_cols:
                    break
                rs = int(td.get("rowspan", 1) or 1)
                cs = int(td.get("colspan", 1) or 1)
                value = _norm_text(td.get_text(" "))
                for r2 in range(ri, min(rows, ri + rs)):
                    for c2 in range(ci, min(max_cols, ci + cs)):
                        grid[r2][c2] = value
                ci += cs
        # None → 빈 문자열
        clean = [["" if v is None else v for v in row] for row in grid]
        out.append((_normalize_section_label(_section_for(table)), clean))
    return out


def write_wide_csv_per_day(
    grids: list[tuple[str, list[list[str]]]],
    *,
    title: str,
    data_date: date,
    out_path: Path,
) -> Path:
    """일자별 보고서 형태 wide CSV — 헤더 + 6개 표를 위→아래로 그대로 복원."""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow([title])
        w.writerow([f"data_date,{data_date.isoformat()}"])
        for section, grid in grids:
            if not grid:
                continue
            # 빈(헤더만 있는) 첫 표는 그대로 한 줄짜리 — 그래도 기록
            w.writerow([])
            label = f"[{section}]" if section else "[표]"
            w.writerow([label])
            for row in grid:
                w.writerow(row)
    return out_path


# ---- Section 시계열 (행=항목, 열=일자 ‘일자(컬럼)’) ----------------------


# 보고서가 가지는 “기준 컬럼”들 — '24말 / '25말 / 26.3말 같은 것들은 일자별로
# 그날 시점에서만 의미가 있고, 일자 시계열에는 “해당 보고일의 당일 시세”만 모은다.
# 한 표에서 “당일 데이터” 컬럼은 보통 마지막 ‘일자 형태’ 컬럼(예: 4/30)이다.
_DATE_COL_RGX = re.compile(r"^(\d{1,2})\s*[/.]\s*(\d{1,2})$")


def _pick_today_column_idx(header_row: list[str], data_date: date) -> int | None:
    """헤더 행에서 'M/D' 또는 'M.D' 형식 중 data_date 와 가장 가까운 열 인덱스를 반환.

    완전 매칭이 없으면 마지막 'M/D' 컬럼을 사용. 그래도 없으면 None.
    """

    candidates: list[tuple[int, int, int]] = []  # (col, month, day)
    for i, h in enumerate(header_row):
        m = _DATE_COL_RGX.match((h or "").strip())
        if not m:
            continue
        candidates.append((i, int(m.group(1)), int(m.group(2))))
    if not candidates:
        return None
    # 정확히 데이터 일자와 일치하는 컬럼이 있으면 그것
    for col, mo, da in candidates:
        if mo == data_date.month and da == data_date.day:
            return col
    # 없으면 마지막 일자 컬럼(보통 보고일)을 사용
    return candidates[-1][0]


def extract_today_values(
    grids: list[tuple[str, list[list[str]]]],
    *,
    data_date: date,
) -> dict[str, dict[str, str]]:
    """그날 보고서에서 (섹션 → {항목: 당일 값}) 사전을 만든다.

    예) {'금리': {'콜금리(1일,%)': '2.51', 'CD(91일,%)': '2.81', ...}, '주가': {...}, ...}
    """

    sections: dict[str, dict[str, str]] = {}
    for section, grid in grids:
        if not grid or not section:
            continue
        if len(grid) < 2 or len(grid[0]) < 2:
            continue
        header_row = grid[0]
        col = _pick_today_column_idx(header_row, data_date)
        if col is None:
            continue
        store: dict[str, str] = {}
        for row in grid[1:]:
            if col >= len(row):
                continue
            label = (row[0] or "").strip()
            if not label:
                continue
            val = (row[col] or "").strip()
            if val == "":
                continue
            # 같은 항목명이 두 번 나오면 첫 번째 값 우선(보통 동일)
            store.setdefault(label, val)
        if store:
            sections.setdefault(section, {}).update(store)
    return sections


# ---- 인덱스 메타 CSV ------------------------------------------------------


_INDEX_FIELDS = [
    "data_date",
    "ntt_id",
    "title",
    "registered_at",
    "atch_file_id",
    "file_sn",
    "file_name",
    "hwp_path",
    "csv_path",
    "table_count",
    "row_count",
    "fetched_at",
]


def _load_index() -> dict[str, dict]:
    """기존 index.csv 를 (data_date → row) 사전으로 로드."""

    if not INDEX_CSV.exists():
        return {}
    out: dict[str, dict] = {}
    with open(INDEX_CSV, "r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            d = (r.get("data_date") or "").strip()
            if d:
                out[d] = r
    return out


def _save_index(index: dict[str, dict]) -> None:
    INDEX_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(index.values(), key=lambda r: r.get("data_date", ""), reverse=True)
    with open(INDEX_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_INDEX_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ---- 메인 처리 함수 -------------------------------------------------------


@dataclass
class CrawlOptions:
    """크롤 동작 옵션."""

    start_page: int = 1
    end_page: int = 651
    stop_year: int = 2015  # 이 연도까지(포함) 받는다. 더 이전이면 중단.
    overwrite: bool = False  # 이미 다운로드된 일자도 다시 가져올지
    sleep_per_request: float = 0.4
    sleep_per_page: float = 0.6
    fetch_attachments_max: int = 1
    items: list[dict] = field(default_factory=list)


def process_post(
    session: requests.Session,
    item: ListItem,
    *,
    overwrite: bool,
    series: dict[str, dict[str, dict[str, str]]] | None = None,
) -> dict | None:
    """한 게시물을 받아 HWP→HTML→CSV(3종) 까지 처리하고 메타 dict 반환.

    `series` 가 주어지면 그날 “당일 시세 컬럼”에서 뽑은 (섹션→항목→{날짜:값})
    매핑을 누적해, crawl 루프가 끝난 뒤 섹션별 시계열 CSV 를 만들 수 있다.
    """

    if item.data_date is None:
        logger.info("skip: title has no data date — %s", item.title)
        return None

    ymd = item.data_date.strftime("%Y%m%d")
    hwp_path = HWP_DIR / f"{ymd}.hwp"
    html_dir = HTML_DIR / ymd
    csv_path = CSV_DIR / f"{ymd}.csv"
    wide_path = WIDE_DIR / f"{ymd}.csv"

    if (
        not overwrite
        and csv_path.exists()
        and hwp_path.exists()
        and wide_path.exists()
    ):
        logger.info("skip existing %s", ymd)
        return None

    attachments = fetch_detail_attachments(session, item.ntt_id)
    if not attachments:
        logger.warning("no attachment for %s (%s)", item.title, item.ntt_id)
        return None
    att = attachments[0]

    logger.info("get %s — %s", ymd, item.title)
    download_attachment(session, att, hwp_path)
    try:
        html_path = hwp_to_html(hwp_path, html_dir, timeout=60.0)
    except Exception as exc:
        logger.warning("hwp→html failed %s: %s", ymd, exc)
        return None
    if not html_path.exists():
        logger.warning("html convert failed for %s", ymd)
        return None

    source_url = (
        f"{DETAIL_URL}?bbsId={BBS_ID}&searchNttId1={item.ntt_id}&menuNo={MENU_NO}"
    )

    rows = parse_tables_to_rows(html_path, data_date=item.data_date, source_url=source_url)
    write_csv(rows, csv_path)
    table_count = len({r["table_idx"] for r in rows})

    grids = parse_tables_to_grids(html_path)
    write_wide_csv_per_day(
        grids,
        title=item.title,
        data_date=item.data_date,
        out_path=wide_path,
    )

    if series is not None:
        today = extract_today_values(grids, data_date=item.data_date)
        d_iso = item.data_date.isoformat()
        for section, items in today.items():
            sec_store = series.setdefault(section, {})
            for label, val in items.items():
                sec_store.setdefault(label, {})[d_iso] = val

    return {
        "data_date": item.data_date.isoformat(),
        "ntt_id": item.ntt_id,
        "title": item.title,
        "registered_at": item.registered_at,
        "atch_file_id": att.atch_file_id,
        "file_sn": att.file_sn,
        "file_name": att.file_name,
        "hwp_path": str(hwp_path),
        "csv_path": str(csv_path),
        "wide_path": str(wide_path),
        "table_count": table_count,
        "row_count": len(rows),
        "fetched_at": datetime.now().isoformat(timespec="seconds"),
    }


def rebuild_wide_csvs(
    *,
    overwrite: bool = True,
    only_dates: Iterable[str] | None = None,
) -> dict:
    """이미 변환되어 있는 모든 xhtml 을 다시 읽어 wide/시계열 CSV 를 만든다.

    출력:
      data/mofe_daily/csv/wide/<YYYYMMDD>.csv  — 일자별 보고서 형태
      data/mofe_daily/wide/<섹션>.csv           — 섹션별 시계열
    """

    WIDE_DIR.mkdir(parents=True, exist_ok=True)
    SERIES_DIR.mkdir(parents=True, exist_ok=True)

    # 인덱스 로드 — title, data_date 매핑용
    index = _load_index()
    if not index:
        # 인덱스가 없으면 csv 폴더에서 일자만이라도 끌어온다
        for p in CSV_DIR.glob("*.csv"):
            stem = p.stem
            if len(stem) == 8 and stem.isdigit():
                d = f"{stem[0:4]}-{stem[4:6]}-{stem[6:8]}"
                index.setdefault(d, {"data_date": d, "title": "", "ntt_id": ""})

    # only_dates 필터(YYYY-MM-DD 형식)
    only_set: set[str] | None = None
    if only_dates:
        only_set = {str(d).strip() for d in only_dates if str(d).strip()}

    # (섹션, 항목) → {data_date: value}
    series: dict[str, dict[str, dict[str, str]]] = {}

    saved = 0
    skipped = 0
    failed = 0
    sorted_keys = sorted(index.keys())  # data_date 오름차순
    for d_iso in sorted_keys:
        if only_set and d_iso not in only_set:
            continue
        meta = index[d_iso]
        try:
            data_date = date.fromisoformat(d_iso)
        except ValueError:
            continue
        ymd = data_date.strftime("%Y%m%d")
        html_path = HTML_DIR / ymd / "index.xhtml"
        if not html_path.exists():
            logger.warning("html missing %s", html_path)
            failed += 1
            continue

        wide_path = WIDE_DIR / f"{ymd}.csv"
        if not overwrite and wide_path.exists():
            skipped += 1
            # 시계열은 다시 빌드하기 위해 grid 추출은 그대로 진행
        try:
            grids = parse_tables_to_grids(html_path)
            if not grids:
                logger.warning("no tables for %s", d_iso)
                failed += 1
                continue
            title = meta.get("title") or f"일일경제지표 {d_iso}"
            write_wide_csv_per_day(
                grids,
                title=title,
                data_date=data_date,
                out_path=wide_path,
            )
            saved += 1
            today = extract_today_values(grids, data_date=data_date)
            for section, items in today.items():
                sec_store = series.setdefault(section, {})
                for label, val in items.items():
                    sec_store.setdefault(label, {})[d_iso] = val
        except Exception as exc:
            logger.warning("rebuild failed %s: %s", d_iso, exc)
            failed += 1

    # 섹션별 시계열 wide CSV 작성
    for section, items in series.items():
        # 모든 일자 합집합
        all_dates = sorted({d for vals in items.values() for d in vals.keys()})
        # 섹션 이름에서 파일명 안전화
        safe = re.sub(r"[\\/:*?\"<>|]+", "_", section).strip() or "표"
        out = SERIES_DIR / f"{safe}.csv"
        with open(out, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["항목"] + all_dates)
            for label in sorted(items.keys()):
                row = [label] + [items[label].get(d, "") for d in all_dates]
                w.writerow(row)

    return {
        "wide_per_day": saved,
        "skipped_per_day": skipped,
        "failed": failed,
        "section_files": sorted(series.keys()),
    }


def _write_section_series(
    series: dict[str, dict[str, dict[str, str]]],
) -> list[str]:
    """누적된 (섹션→항목→{날짜:값}) 사전을 섹션별 시계열 wide CSV 로 저장."""

    SERIES_DIR.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for section, items in series.items():
        if not items:
            continue
        all_dates = sorted({d for vals in items.values() for d in vals.keys()})
        safe = re.sub(r"[\\/:*?\"<>|]+", "_", section).strip() or "표"
        out = SERIES_DIR / f"{safe}.csv"
        with open(out, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["항목"] + all_dates)
            for label in sorted(items.keys()):
                row = [label] + [items[label].get(d, "") for d in all_dates]
                w.writerow(row)
        written.append(str(out))
    return written


def crawl(options: CrawlOptions) -> dict:
    """목록 → 상세 → 다운로드 → 변환 → CSV(3종) 통합 루프."""

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    WIDE_DIR.mkdir(parents=True, exist_ok=True)
    index = _load_index()
    saved_count = 0
    skipped_count = 0
    fail_count = 0
    series: dict[str, dict[str, dict[str, str]]] = {}

    session = requests.Session()

    for page in range(options.start_page, options.end_page + 1):
        try:
            items = fetch_list_page(session, page)
        except Exception as exc:
            logger.warning("list page %d failed: %s", page, exc)
            time.sleep(2)
            continue
        if not items:
            logger.info("page %d empty — stop", page)
            break

        # 페이지 안의 가장 오래된 일자 확인 — stop_year 보다 이전 항목만 있으면 중단
        page_dates = [it.data_date for it in items if it.data_date]
        oldest = min(page_dates) if page_dates else None
        logger.info(
            "page %d items=%d oldest=%s newest=%s",
            page,
            len(items),
            oldest.isoformat() if oldest else "?",
            max(page_dates).isoformat() if page_dates else "?",
        )

        for item in items:
            if item.data_date and item.data_date.year < options.stop_year:
                continue
            try:
                meta = process_post(
                    session,
                    item,
                    overwrite=options.overwrite,
                    series=series,
                )
            except Exception as exc:
                logger.warning("post failed %s (%s): %s", item.title, item.ntt_id, exc)
                fail_count += 1
                meta = None
            if meta is None:
                skipped_count += 1
            else:
                index[meta["data_date"]] = meta
                saved_count += 1
                _save_index(index)
            time.sleep(options.sleep_per_request)

        # 매 페이지가 끝날 때 시계열 wide CSV 도 함께 갱신
        # (중간 종료해도 그 시점까지의 시계열은 디스크에 남는다)
        _write_section_series(series)

        if oldest is not None and oldest.year < options.stop_year:
            logger.info("reached stop_year=%d, finish", options.stop_year)
            break
        time.sleep(options.sleep_per_page)

    written_series = _write_section_series(series)

    return {
        "saved": saved_count,
        "skipped": skipped_count,
        "failed": fail_count,
        "index_csv": str(INDEX_CSV),
        "section_series_csvs": written_series,
    }
