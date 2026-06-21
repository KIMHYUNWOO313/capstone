import base64
import html
import re
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.pdfmetrics import registerFont
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


TITLE_FONT = "HYGothic-Medium"
BODY_FONT = "HYSMyeongJo-Medium"


def _register_fonts() -> None:
    try:
        registerFont(UnicodeCIDFont(TITLE_FONT))
        registerFont(UnicodeCIDFont(BODY_FONT))
    except Exception:
        pass


def _text(value, fallback="-") -> str:
    value = "" if value is None else str(value)
    value = re.sub(r"\s+", " ", value).strip()
    return value or fallback


def _paragraph_text(value: str) -> str:
    escaped = html.escape(str(value or "").strip())
    return escaped.replace("\n", "<br/>") or "-"


def _image_from_data_url(data_url: str, max_width: float, max_height: float) -> Image | None:
    if not data_url or "," not in data_url:
        return None
    try:
        raw = base64.b64decode(data_url.split(",", 1)[1])
        bio = BytesIO(raw)
        image = Image(bio)
        ratio = min(max_width / image.imageWidth, max_height / image.imageHeight)
        image.drawWidth = image.imageWidth * ratio
        image.drawHeight = image.imageHeight * ratio
        image.hAlign = "CENTER"
        return image
    except Exception:
        return None


def build_report_pdf(payload: dict) -> bytes:
    _register_fonts()
    buffer = BytesIO()
    page_width, _ = A4
    margin = 14 * mm
    content_width = page_width - (margin * 2)

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=margin,
        rightMargin=margin,
        topMargin=12 * mm,
        bottomMargin=12 * mm,
        title=_text(payload.get("title"), "AI 가격 예측 리포트"),
    )

    styles = {
        "kicker": ParagraphStyle(
            "kicker",
            fontName=TITLE_FONT,
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#d6e887"),
            alignment=TA_LEFT,
        ),
        "title": ParagraphStyle(
            "title",
            fontName=TITLE_FONT,
            fontSize=19,
            leading=25,
            textColor=colors.HexColor("#fff6c7"),
            alignment=TA_LEFT,
        ),
        "subtitle": ParagraphStyle(
            "subtitle",
            fontName=BODY_FONT,
            fontSize=9,
            leading=14,
            textColor=colors.HexColor("#f3edc8"),
            alignment=TA_LEFT,
        ),
        "section": ParagraphStyle(
            "section",
            fontName=TITLE_FONT,
            fontSize=14,
            leading=18,
            textColor=colors.HexColor("#17230f"),
            alignment=TA_LEFT,
        ),
        "body": ParagraphStyle(
            "body",
            fontName=BODY_FONT,
            fontSize=9.5,
            leading=15,
            textColor=colors.HexColor("#25351f"),
            alignment=TA_LEFT,
        ),
        "center": ParagraphStyle(
            "center",
            fontName=TITLE_FONT,
            fontSize=9,
            leading=13,
            textColor=colors.HexColor("#25351f"),
            alignment=TA_CENTER,
        ),
    }

    title = _text(payload.get("title"), "농산물 가격 예측 분석 리포트")
    subtitle = _text(payload.get("subtitle"), "실측 가격, 구간 예측(10일) 범위, 점예측(3일)을 기반으로 작성된 자동 리포트입니다.")
    item = _text(payload.get("item"))
    model = _text(payload.get("model"), "구간 예측(10일) + 점예측(3일)")
    created_at = _text(payload.get("created_at"))
    analysis = str(payload.get("analysis") or "분석 내용이 없습니다.")
    summary = payload.get("summary") or {}

    story = []
    cover_rows = [
        [
            Paragraph("AGRICULTURAL PRICE FORECAST REPORT", styles["kicker"]),
        ],
        [Paragraph(_paragraph_text(title), styles["title"])],
        [Paragraph(_paragraph_text(subtitle), styles["subtitle"])],
    ]
    cover = Table(cover_rows, colWidths=[content_width], hAlign="LEFT")
    cover.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#123516")),
        ("BOX", (0, 0), (-1, -1), 0, colors.HexColor("#123516")),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 14),
        ("RIGHTPADDING", (0, 0), (-1, -1), 14),
    ]))
    story.append(cover)

    meta = Table(
        [
            ["품목", "모델", "작성 시각"],
            [item, model, created_at],
        ],
        colWidths=[content_width / 3] * 3,
    )
    meta.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#edf3d7")),
        ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#f8faee")),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#17230f")),
        ("FONTNAME", (0, 0), (-1, -1), TITLE_FONT),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#d8dfc7")),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.extend([Spacer(1, 7 * mm), meta, Spacer(1, 8 * mm)])

    story.append(Paragraph("01  가격 예측 그래프", styles["section"]))
    story.append(Spacer(1, 3 * mm))
    chart = _image_from_data_url(str(payload.get("chart_image") or ""), content_width - 8 * mm, 78 * mm)
    if chart:
        story.append(chart)
    else:
        story.append(Paragraph("그래프 이미지를 생성하지 못했습니다.", styles["body"]))
    story.append(Spacer(1, 8 * mm))

    story.append(Paragraph("02  GPT 결과 분석", styles["section"]))
    story.append(Spacer(1, 3 * mm))
    analysis_box = Table(
        [[Paragraph(_paragraph_text(analysis), styles["body"])]],
        colWidths=[content_width],
    )
    analysis_box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f7f9eb")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#d8dfc7")),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(analysis_box)
    story.append(Spacer(1, 8 * mm))

    story.append(Paragraph("03  예측 요약", styles["section"]))
    story.append(Spacer(1, 3 * mm))
    summary_table = Table(
        [
            ["최근 실측가", "점예측(3일)", "구간 예측(10일) 범위"],
            [
                _text(summary.get("current_price")),
                _text(summary.get("timesfm_price")),
                _text(summary.get("chronos_range")),
            ],
        ],
        colWidths=[content_width / 3] * 3,
    )
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#edf3d7")),
        ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#fffdf4")),
        ("FONTNAME", (0, 0), (-1, -1), TITLE_FONT),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#d8dfc7")),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(summary_table)

    doc.build(story)
    return buffer.getvalue()
