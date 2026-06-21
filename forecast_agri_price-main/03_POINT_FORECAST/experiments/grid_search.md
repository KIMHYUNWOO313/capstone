# Grid Search — TimesFM LoRA / IBM TTM Fine-tune 하이퍼파라미터 탐색

> 본 문서는 `[4way_comparison.md](4way_comparison.md)` 의 TimesFM LoRA / TTM FT 항목이 어떤 그리드 탐색을 거쳐 best 셋팅을 선정했는지 정리합니다.

---

## 1. TimesFM 2.5 LoRA 그리드

원본: [`grid_timesfm.csv`](grid_timesfm.csv)

| trial | r | α | lr | num_samples | epochs | val_loss | elapsed |
|---|---:|---:|---:|---:|---:|---:|---:|
| lora_r4_a8_lr1e4 | 4 | 8 | 1e-4 | 2000 | 3 | 0.611 | 3.3 min |
| **★ lora_r8_a16_lr1e4** | **8** | **16** | **1e-4** | 2000 | 3 | **0.583** ← best | 3.3 min |
| lora_r16_a32_lr5e5 | 16 | 32 | 5e-5 | 2000 | 3 | 0.608 | 3.3 min |

**선정 셋팅**: `r=8, α=16, lr=1e-4, epochs=3, num_samples=2000`
- val_loss 0.583 (최저)
- r=4 보다 표현력 충분, r=16 보다 과적합 위험 낮음
- lr=1e-4 가 lr=5e-5 보다 빠른 수렴

**4-way 평가 결과** (test set): MASE 0.811 → ZS 0.806 대비 +0.6% 악화. **LoRA 가 ZS 를 능가 못 함을 확인**.

---

## 2. IBM TTM Fine-tune 그리드

원본: [`grid_ttm.csv`](grid_ttm.csv)

| trial | revision | lr | epochs | val_loss | elapsed |
|---|---|---:|---:|---:|---:|
| r2_main_lr1e3_e8 | main | 1e-3 | 8 | 0.272 | 1.9 min |
| **★ r2_main_lr5e4_e12** | **main** | **5e-4** | **12** | **0.246** ← best | 2.2 min |
| r2_main_lr2e3_e6 | main | 2e-3 | 6 | 0.274 | 1.8 min |

**선정 셋팅**: `revision=main, lr=5e-4, epochs=12` (backbone freeze + decoder/head 학습)
- val_loss 0.246 (최저)
- lr 너무 높으면 (1e-3, 2e-3) 발산 경향
- epochs 12 가 8 보다 안정적 수렴

**4-way 평가 결과** (test set): MASE 0.931 → ZS 2.184 대비 -57.4% 개선. **TTM 의 큰 폭 개선은 base 약함 (channel mixer random init) 의 정상화** 효과 (자세한 인과 [`4way_comparison.md`](4way_comparison.md) §4 발견 3).

---

## 3. 시도되지 않은 그리드 (한계점)

- **TimesFM 의 더 긴 context** — 본 프로젝트 384일. 모델 최대 16,384까지 지원. 1024~1536 시도 시 추가 개선 가능성
- **TTM-R2.1 (일/주 데이터 특화)** — `freq_token` 주입 필요로 R2 main 만 사용. R2.1 시도 시 일별 데이터에 더 적합 가능성
- **TimesFM LoRA 의 더 큰 데이터·더 많은 epoch** — 현 2000 samples × 3 epoch. 본 데이터 규모 (46 시리즈) 에선 ROI 불확실
- **TimesFM LoRA target 모듈 변경** — 현재 attention QKVO 만. MLP 까지 포함 시 변화 가능성

---

## 4. 산출물 매핑

| 산출물 | 위치 |
|---|---|
| TimesFM LoRA 그리드 로그 | [`grid_timesfm.csv`](grid_timesfm.csv) |
| IBM TTM 그리드 로그 | [`grid_ttm.csv`](grid_ttm.csv) |
