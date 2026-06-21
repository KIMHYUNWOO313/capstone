"""산지 영문명 → 기상특보 광역 stnId 매핑.

기상특보(WthrWrnInfoService)는 광역 시·도 단위로 발효된다.
meta_baseline.json의 item_station_map은 시군 영문명(예: "jecheon")으로 되어 있으므로
광역 단위 stnId로 변환해야 한다.

stnId 출처: 기상청 "기상특보 조회서비스 Open API 활용가이드" 첨부 "지점코드" 표.
세부 코드가 다른 경우 docx를 참고해 보정하세요.
"""

STATION_TO_REGION_STN: dict[str, int] = {
    # 강원
    "gangneung":      105,
    "daegwallyeong":  105,
    # 경기
    "icheon":         109,
    # 충북
    "jecheon":        131,
    # 충남 / 세종 / 대전
    "cheonan":        133,
    "geumsan":        133,
    # 전북
    "jeonju":         146,
    # 전남 / 광주
    "haenam":         156,
    "mokpo":          156,
    # 경북 / 대구
    "yeongju":        143,
    "andong":         143,
    "sangju":         143,
    "pohang":         143,
    # 경남 / 부산 / 울산
    "changwon":       159,
    "sancheong":      159,
    "miryang":        159,
    # 제주
    "jeju":           184,
    "seongsan":       184,
}

STN_TO_REGION_KR: dict[int, str] = {
    105: "강원",
    109: "경기",
    131: "충북",
    133: "충남·세종·대전",
    143: "경북·대구",
    146: "전북",
    156: "전남·광주",
    159: "경남·부산·울산",
    184: "제주",
}

STATION_KR: dict[str, str] = {
    "gangneung":     "강릉",
    "daegwallyeong": "대관령",
    "icheon":        "이천",
    "jecheon":       "제천",
    "cheonan":       "천안",
    "geumsan":       "금산",
    "jeonju":        "전주",
    "haenam":        "해남",
    "mokpo":         "목포",
    "yeongju":       "영주",
    "andong":        "안동",
    "sangju":        "상주",
    "pohang":        "포항",
    "changwon":      "창원",
    "sancheong":     "산청",
    "miryang":       "밀양",
    "jeju":          "제주",
    "seongsan":      "성산",
}


def stations_to_region_stns(stations: list[str]) -> dict[int, list[str]]:
    """광역 stnId → 그 광역에 속한 산지 영문명 리스트(dedupe)."""
    out: dict[int, list[str]] = {}
    for st in set(stations):
        stn = STATION_TO_REGION_STN.get(st)
        if stn is None:
            continue
        out.setdefault(stn, []).append(st)
    return out
