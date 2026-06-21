# User Prediction - 게임 사용자 수 예측 대시보드

Django, HTML, CSS, JavaScript로 구현한 게임 사용자 수 예측 웹 애플리케이션입니다.

## 기능

- **홈페이지**: User Prediction 타이틀, Get Started 버튼, 반투명 애니메이션 배경 (게임 아이콘, 패치노트/리뷰 더미)
- **대시보드**:
  - 패치노트 입력 및 목록
  - 사용자 수 예측 (AI 모델 연동 예정)
  - AWS EC2 인스턴스 관리 (t2, t3 시리즈)
  - 사용자 수 기반 스케일링 권장 (AI 연동 예정)
- **다크/라이트 모드**: 우측 상단 아이콘으로 전환

## 실행 방법

```powershell
# 가상환경 활성화
.\venv\Scripts\Activate.ps1

# 서버 실행
python manage.py runserver
```

브라우저에서 http://127.0.0.1:8000/ 접속

## AWS와 게임 업계

많은 게임 회사들이 AWS를 사용합니다. EC2 t2, t3 시리즈는 비용 효율적인 범용 인스턴스로, 소규모~중규모 게임 서버에 적합합니다.
