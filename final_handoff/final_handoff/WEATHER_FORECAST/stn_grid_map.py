"""산지 영문명 → 기상청 단기예보 격자(nx, ny) / 중기기온예보 구역코드(regId) 매핑.

추출 출처:
- nx/ny: 기상청41_단기예보..격자_위경도.xlsx (시청·군청 소재지 기준)
- regId: 중기예보_중기기온예보구역코드_2025.12.xlsx (시·군 단위 기온 구역)
"""

# 단기예보 (getVilageFcst) 격자 좌표 (nx, ny)
STATION_TO_GRID: dict[str, tuple[int, int]] = {
    "jecheon":       (81, 118),   # 충북 제천시
    "yeongju":       (89, 111),   # 경북 영주시
    "andong":        (91, 106),   # 경북 안동시
    "icheon":        (68, 121),   # 경기 이천시
    "cheonan":       (63, 110),   # 충남 천안시 동남구
    "geumsan":       (69,  95),   # 충남 금산군
    "jeonju":        (63,  89),   # 전북 전주시 완산구
    "haenam":        (54,  61),   # 전남 해남군
    "mokpo":         (50,  67),   # 전남 목포시
    "sangju":        (81, 102),   # 경북 상주시
    "pohang":        (102, 94),   # 경북 포항시 남구
    "changwon":      (90,  77),   # 경남 창원시 의창구
    "sancheong":     (76,  80),   # 경남 산청군
    "miryang":       (92,  83),   # 경남 밀양시
    "jeju":          (53,  38),   # 제주 제주시
    "seongsan":      (60,  37),   # 제주 서귀포시 성산읍
    "gangneung":     (92, 131),   # 강원 강릉시
    "daegwallyeong": (89, 130),   # 강원 평창군 대관령면
}

# 중기기온예보 (getMidTa) 시·군 구역코드 (regId)
STATION_TO_MID_REGID: dict[str, str] = {
    "jecheon":       "11C10201",
    "yeongju":       "11H10401",
    "andong":        "11H10501",
    "icheon":        "11B20701",
    "cheonan":       "11C20301",
    "geumsan":       "11C20601",
    "jeonju":        "11F10201",
    "haenam":        "11F20302",
    "mokpo":         "21F20801",
    "sangju":        "11H10302",
    "pohang":        "11H10201",
    "changwon":      "11H20301",
    "sancheong":     "11H20703",
    "miryang":       "11H20601",
    "jeju":          "11G00201",
    "seongsan":      "11G00101",
    "gangneung":     "11D20501",
    "daegwallyeong": "11D20201",
}

# 한글 표기 (보고서·로그용)
STATION_KR: dict[str, str] = {
    "jecheon":       "제천",
    "yeongju":       "영주",
    "andong":        "안동",
    "icheon":        "이천",
    "cheonan":       "천안",
    "geumsan":       "금산",
    "jeonju":        "전주",
    "haenam":        "해남",
    "mokpo":         "목포",
    "sangju":        "상주",
    "pohang":        "포항",
    "changwon":      "창원",
    "sancheong":     "산청",
    "miryang":       "밀양",
    "jeju":          "제주",
    "seongsan":      "성산",
    "gangneung":     "강릉",
    "daegwallyeong": "대관령",
}


def assert_complete():
    """세 사전의 키 집합이 동일한지 검증."""
    a = set(STATION_TO_GRID)
    b = set(STATION_TO_MID_REGID)
    c = set(STATION_KR)
    assert a == b == c, f"키 불일치: grid={a^b}, kr={a^c}"
    return len(a)


if __name__ == "__main__":
    n = assert_complete()
    print(f"산지 {n}개, 매핑 일관성 OK")
    for st in sorted(STATION_TO_GRID):
        nx, ny = STATION_TO_GRID[st]
        reg = STATION_TO_MID_REGID[st]
        print(f"  {st:14s} {STATION_KR[st]:4s} nx={nx:3d} ny={ny:3d}  regId={reg}")
