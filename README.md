# 🚚 Vehicle Delivery AI Smart Matching System
### AI 기반 탁송/대리 기사 매칭 서비스 — 차량 흠집 탐지 · 감정 분석 · 최적 경로 추천

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=flat-square&logo=flask&logoColor=white)
![MySQL](https://img.shields.io/badge/MySQL-4479A1?style=flat-square&logo=mysql&logoColor=white)

<br>

탁송 차량과 대리 기사를 AI 기술로 매칭하는 통합 서비스입니다.  
기사 신뢰도 기반 자동 배차, 고객 리뷰 감정 분석, **탁송 전·후 차량 흠집 자동 탐지**, 최적 운행 경로 추천을 통합한 플랫폼입니다.

> 🗓 **개발 기간**: 2026.01.02 – 2026.01.16  
> 👥 **팀 구성**: 4인 팀 프로젝트

---

## 👥 팀 구성 & 역할 분담

| 팀원 | 담당 파트 | 주요 내용 |
|------|----------|----------|
| **강지수 ★** | **차량 흠집 탐지** | ResNet18 기반 CNN 심각도 모델 설계·학습, OpenCV 위치 탐지, 라벨링 체계 설계 |
| 장성우 | 매칭 알고리즘 | 기사 신뢰도 기반 자동 배차, 최적 운행 경로 추천 |
| 강대규 | 리뷰 감정 분석 | 고객 리뷰 키워드 AI 학습, 감정 분류 시스템 |
| 이상혁 | 서버 & DB | MySQL 설계, Flask 서버 구축, 웹 페이지 구현 |

---

## 🏆 나의 담당 — 차량 흠집 탐지 시스템

### 핵심 성과

| 항목 | 내용 |
|------|------|
| 수집 차량 | 126대 직접 수집 (인터넷 이미지 + 스크래치 합성) |
| 심각도 라벨 | 0~4단계 직접 설계 |
| 차량당 촬영 | 전·후 각 6앵글 (12장/대) |
| CNN 학습 | 67대, 25 epoch, Loss 0.638 → 0.018 |
| 실제 추론 결과 | 차량 26_2 → severity: 4, raw_score: 4.0 ✅ |
| 탐지 방식 | CNN(심각도) + OpenCV(위치) 이중 파이프라인 |

---

## 📌 프로젝트 개요

차량 탁송 서비스에서는 아래와 같은 문제가 발생합니다.

- 기사와 화물 매칭이 전화·문자 등 **수작업 중심**으로 이루어짐
- 탁송 중 발생한 **차량 손상 책임 분쟁**
- 배차 지연, 경로 비효율, 수익 불투명

본 프로젝트는 이를 해결하기 위해 **AI와 데이터를 기반으로 탁송 매칭 과정을 자동화**했습니다.

---

## 🧠 주요 기능

### 🚗 차량 흠집 탐지 (내가 담당한 파트)

탁송 전·후 차량 사진을 비교하여 **신규 흠집 여부 및 손상 심각도(0~4)**를 자동 판정합니다.

**두 가지 모듈로 구성:**

#### ① CNN 회귀 모델 — 심각도(0~4) 예측
- ResNet18 기반 **9채널 입력 직접 설계** (before 3ch + after 3ch + diff 3ch)
- 67대 학습 / 25 epoch / Loss 0.638 → 0.018 수렴
- 6각도 뷰 예측값을 **max+mean 가중 풀링**으로 최종 점수 산출
- 결과 JSON 저장 + 엑셀 사고점수 자동 반영

#### ② OpenCV 파이프라인 — 흠집 위치 시각화
```
탁송 전 이미지 + 탁송 후 이미지
        │
        ▼
① ECC 정렬          촬영 각도 차이 자동 보정
        │
        ▼
② CLAHE 명암 보정   조명 차이 제거
        │
        ▼
③ 픽셀 차분         변화 영역 검출 (absdiff)
        │
        ▼
④ 형태학적 필터링   노이즈 제거
        │
        ▼
⑤ 오탐 필터         위치·크기·종횡비 기반 정확도 향상
        │
        ▼
⑥ 바운딩 박스       초록 박스로 손상 위치 시각화
```

#### 심각도 라벨링 체계 (0~4단계 직접 설계)

| 단계 | 수준 | 기준 |
|:---:|------|------|
| **0** | 변화 없음 | 신규 흠집 없음 |
| **1** | 경미 | 얕은 스크래치, 멀리서 잘 안 보임 |
| **2** | 보통 | 육안으로 명확, 도장 손상, 부분 도색 가능 |
| **3** | 심각 | 넓거나 깊음, 찌그러짐, 수리 필요 |
| **4** | 매우 심각 | 구조적 손상 수준, 즉시 수리·교체 필요 |

> 손상 크기·깊이·부위 가중치를 반영한 **결정 규칙** 직접 설계

#### 데이터셋 구성

| 심각도 | Score 0 | Score 1 | Score 2 | Score 3 | Score 4 |
|:------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| 대수 | 46대 | 22대 | 23대 | 21대 | 14대 |

#### 실제 추론 결과 (차량 26_2)

```json
{
  "index": "26_2",
  "severity": 4,
  "raw_score": 4.0,
  "diff_mean": 0.007066433007518451,
  "views_used": ["front", "rear", "left", "right", "left_side", "right_side"],
  "missing_views": []
}
```

---

### 🧭 기사 매칭 시스템 (Driver Matching)

주문 위치와 기사 위치 데이터를 기반으로 **최적 기사를 자동 매칭**합니다.

- 거리·시간·수익성을 고려한 스코어링
- 조건 필터링 후 우선순위 추천
- 경로 지도 화면 표시

---

### 💬 리뷰 감정 분석 (Review Sentiment Analysis)

고객 리뷰 텍스트를 분석하여 **긍정/부정 감정을 분류**, 기사 신뢰도 점수에 반영합니다.

- 리뷰 텍스트 형태소 분석
- 정수로 인코딩 후 모델 구성, 학습으로 긍정/부정 분류
- 신뢰 확률 기반 점수 가산·감산

---

## 📂 프로젝트 구조

```
vehicle-delivery-ai-system/
│
├── damage_detection/              # 🚗 차량 흠집 탐지 (강지수 담당)
│   ├── data/
│   ├── notebooks/
│   │   ├── severity.ipynb         # CNN 회귀 모델 학습 & 추론
│   │   └── damege_box.ipynb       # OpenCV 흠집 위치 탐지
│   ├── model/
│   │   └── checkpoints/
│   │       └── regressor_view_mixpool.pth
│   └── result/
│
├── driver_matching/               # 🧭 기사 매칭 / 경로 최적화
│   ├── data/
│   │   ├── delivery_orders_2026.csv
│   │   └── delivery_orders_2026_utf8_version.csv
│   └── notebooks/
│       └── driver_matching_model.ipynb
│
├── review_sentiment/              # 💬 리뷰 감정 분석
│   ├── data/
│   │   ├── drivers_2026.xlsx
│   │   └── review_label.csv
│   └── notebooks/
│       └── review_sentiment_analysis.ipynb
│
├── database/
│   └── mysql/
│
├── docs/
│   ├── 탁송흐름도.png
│   └── 시스템설명.pdf
│
└── README.md
```

---

## 🛠 Tech Stack

| 분류 | 기술 |
|------|------|
| 언어 | Python |
| CNN 모델 | PyTorch, torchvision (ResNet18) |
| 이미지 처리 | OpenCV, PIL |
| 데이터 분석 | Pandas, NumPy |
| 시각화 | Matplotlib, Seaborn |
| 서버 | Flask |
| DB | MySQL |
| 기타 | Jupyter Notebook |

---

## ⚠️ 한계 및 개선 방향

| 한계 | 개선 방향 |
|------|----------|
| 126대 소규모 데이터셋 — Score 3·4(심각) 케이스 희귀 | 데이터 증강, 실제 탁송 이미지 추가 수집 |
| CNN과 OpenCV 모듈 간 통합 연동 미완성 | 심각도 점수 + 위치 정보를 하나의 파이프라인으로 통합 |
| 로컬 서버(Flask) 기반 — 실시간 서비스 환경 미반영 | AWS / GCP 클라우드 배포, YOLO 기반 객체 탐지 통합 |

---

## 📝 One-line Summary

> **탁송 차량의 전·후 이미지를 CNN + OpenCV 이중 파이프라인으로 분석하여 흠집 심각도(0~4)를 자동 판정하고, 기사 매칭·리뷰 감정 분석을 통합한 AI 기반 탁송 서비스를 구현**한 팀 프로젝트입니다.

---

## 👤 개발자

**강지수 (Kang Ji Soo)**  
📧 rkdwl3264@naver.com  
🐙 [github.com/rkdwltn1211](https://github.com/rkdwltn1211)
