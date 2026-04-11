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
| **강지수 ★** | **차량 흠집 탐지** | EfficientNet-B0 기반 CNN 심각도 모델 설계·학습, OpenCV 데이터 파이프라인, 라벨링 체계 설계, FastAPI 서버 + AWS EC2 배포 |
| 장성우 | 매칭 알고리즘 | 기사 신뢰도 기반 자동 배차, 최적 운행 경로 추천 |
| 강대규 | 리뷰 감정 분석 | 고객 리뷰 키워드 AI 학습, 감정 분류 시스템 |
| 이상혁 | 서버 & DB | MySQL 설계, Flask 서버 구축, 웹 페이지 구현 |

---

## 🏆 나의 담당 — 차량 흠집 탐지 시스템

> 💡 팀 프로젝트 이후 담당 파트를 **FastAPI 서버 + AWS EC2 배포**까지 독립 서비스로 고도화했습니다.  
> 👉 **[![GitHub](https://img.shields.io/badge/GitHub-vehicle--damage--detection-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/rkdwltn1211/vehicle-damage-detection)** — 개인 고도화 레포 (Live Demo 포함)

### 핵심 성과

| 항목 | 내용 |
|------|------|
| 수집 차량 | 126대 직접 수집 (인터넷 이미지 + 스크래치 합성) |
| 심각도 라벨 | 0~4단계 직접 설계 |
| 차량당 촬영 | 전·후 각 6앵글 (12장/대) |
| CNN 학습 | EfficientNet-B0, val accuracy 64% |
| 실제 추론 결과 | 차량 26_2 → severity: 4, raw_score: 4.0 ✅ |
| 탐지 방식 | CNN 심각도 판정 + OpenCV 데이터 파이프라인 |
| 배포 | FastAPI + AWS EC2 (t3.micro) |

---

## 📌 프로젝트 개요

차량 탁송 서비스에서는 아래와 같은 문제가 발생합니다.

- 기사와 화물 매칭이 전화·문자 등 **수작업 중심**으로 이루어짐
- 탁송 중 발생한 **차량 손상 책임 분쟁**
- 배차 지연, 경로 비효율, 수익 불투명

본 프로젝트는 이를 해결하기 위해 **AI와 데이터를 기반으로 탁송 매칭 과정을 자동화**했습니다.

---

## 🧠 주요 기능

### 🚗 차량 흠집 탐지 (강지수 담당)

탁송 전·후 차량 사진을 비교하여 **신규 흠집 여부 및 손상 심각도(0~4)**를 자동 판정합니다.

#### CNN 분류 모델 — 심각도(0~4) 예측
- **EfficientNet-B0** (ImageNet pretrained) Transfer Learning
- 클래스 불균형 대응: CrossEntropyLoss 가중치 적용
- Early Stopping (patience=5) 적용
- 데이터 증강으로 634개 → 4,190장 확보
- val accuracy: **64%**

#### 데이터 파이프라인 (OpenCV)
```
before 이미지 + after 이미지
        │
        ▼
① ECC 정렬          촬영 각도 차이 자동 보정
        │
        ▼
② Canny 엣지 탐지   흠집 영역 마스크 생성
        │
        ▼
③ Morphology 필터   노이즈 제거
        │
        ▼
④ ROI 패치 추출     흠집 영역 크롭 → CNN 입력
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
│   │   ├── severity.ipynb         # CNN 모델 학습 & 추론
│   │   └── damege_box.ipynb       # OpenCV 데이터 파이프라인
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
| CNN 모델 | PyTorch, EfficientNet-B0 (torchvision) |
| 이미지 처리 | OpenCV, PIL |
| 데이터 분석 | Pandas, NumPy |
| 시각화 | Matplotlib, Seaborn |
| 백엔드 | FastAPI (강지수 파트), Flask |
| DB | MySQL |
| 배포 | AWS EC2 (강지수 파트) |
| 기타 | Jupyter Notebook |

---

## ⚠️ 한계 및 개선 방향

| 한계 | 개선 방향 |
|------|----------|
| 126대 소규모 데이터셋 — Score 3·4(심각) 케이스 희귀 | 데이터 증강, 실제 탁송 이미지 추가 수집 |
| val accuracy 64% — 소규모 데이터 한계 | CarDD 등 공개 데이터셋 추가 결합 시 성능 향상 가능 |
| 흠집 위치 시각화 미구현 | OpenCV 바운딩 박스 파이프라인 통합 예정 |

---

## 📝 One-line Summary

> **탁송 차량의 전·후 이미지를 EfficientNet-B0 CNN + OpenCV 파이프라인으로 분석하여 흠집 심각도(0~4)를 자동 판정하고, 기사 매칭·리뷰 감정 분석을 통합한 AI 기반 탁송 서비스를 구현**한 팀 프로젝트입니다.

---

## 👤 개발자

**강지수 (Kang Ji Soo)**  
📧 rkdwl3264@naver.com  
🐙 [github.com/rkdwltn1211](https://github.com/rkdwltn1211)
