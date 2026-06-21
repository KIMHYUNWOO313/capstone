# AgriPredict — 농산물 가격 예측 대시보드

Django 기반 웹 애플리케이션으로, **농산물 도매가격**을 단기·중기 관점에서 예측하고 **XAI(설명 가능 AI)** 로 판단 근거를 제공합니다.

## 주요 기능

- **3일 점예측**: TimesFM 2.5 Zero-Shot 기반 단기 가격 전망
- **10일 구간 예측**: Chronos-2 LoRA 기반 P10~P90 확률 구간 및 점 추정
- **XAI 설명**: GPT-4o 기반 판단 근거 (전문가용 / 쉬운 설명 탭)
- **기상·경제 연동**: 기상청·유가·물가 등 covariate를 반영한 예측 컨텍스트
- **실제가 정산(reconcile)**: 외부 API·저장 데이터와 예측값 비교
- **리포트 PDF**: 품목별 예측 결과 요약 다운로드
- **뉴스·품목 페이지**: 농업 뉴스 및 지원 품목 안내

## 기술 스택

| 영역 | 기술 |
|------|------|
| Backend | Django 6, Python 3 |
| Frontend | HTML, CSS, JavaScript, Chart.js |
| 점예측 | TimesFM 2.5 |
| 구간예측 | Chronos-2 (LoRA fine-tuned) |
| XAI | OpenAI GPT-4o / GPT-4o-mini |
| DB (선택) | Firebase Firestore / Django SQLite |
| 배포 | AWS EC2 |

## 프로젝트 구조

```
capstone/
├── config/              # Django 설정
├── dashboard/           # 뷰, API, 예측·XAI·데이터 로직
├── templates/           # 홈, 대시보드, 리포트 UI
├── static/              # CSS, JS, 이미지
├── forecast_agri_price-main/  # ML 파이프라인 (데이터·모델·XAI)
├── xai/                 # XAI 설명 모듈
├── AGRICULTURE/         # 기상·가격 데이터 수집 스크립트
├── data/                # 가공 데이터 (mofe_daily 등)
├── requirements.txt
└── .env.example         # 환경 변수 템플릿
```

## 실행 방법

### 1. 환경 설정

```powershell
# 가상환경 생성 및 활성화
python -m venv venv
.\venv\Scripts\Activate.ps1

# 의존성 설치
pip install -r requirements.txt

# 환경 변수 설정
copy .env.example .env
# .env 파일에 OPENAI_API_KEY, API 키 등 입력
```

### 2. 서버 실행

```powershell
python manage.py migrate
python manage.py runserver
```

브라우저에서 http://127.0.0.1:8000/ 접속

개발 시 `CHRONOS_SKIP_INFERENCE=true` 로 설정하면 GPU 없이 더미 예측으로 UI를 확인할 수 있습니다.

## 환경 변수

`.env.example` 참고. 주요 항목:

| 변수 | 설명 |
|------|------|
| `OPENAI_API_KEY` | XAI 설명 생성용 |
| `FIREBASE_CREDENTIALS_PATH` | Firestore 사용 시 (미설정 시 Django DB만 사용) |
| `CHRONOS_SKIP_INFERENCE` | `true` 시 Chronos 추론 스킵 |
| `WEATHER_API` | 기상청 API |
| `KAMIS_API_KEY` | aT KAMIS 농산물 가격 API |
| `OPINET_API` | 한국석유공사 유가 API |

## API 엔드포인트

| 경로 | 설명 |
|------|------|
| `GET /api/agri/items/` | 예측 가능 품목 목록 |
| `GET /api/agri/chart/` | 차트용 시계열·예측 데이터 |
| `POST /api/agri/run-forecast/` | 예측 실행 |
| `POST /api/agri/explain/` | XAI 설명 생성 |
| `POST /api/agri/reconcile/` | 실제가 정산 |
| `GET /api/report/pdf/` | PDF 리포트 다운로드 |

## 데이터 출처

- **농산물 가격**: aT KAMIS / 농넷 도매시장 통계
- **기상**: 기상청 지상관측·단기/중기 일기예보
- **거시경제**: 한국은행 ECOS, 기획재정부 경제지표
- **유가**: 한국석유공사 Opinet API
- **농업월보**: 농촌진흥청 PDF (XAI 컨텍스트)
