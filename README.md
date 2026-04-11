# 🚚 Vehicle Delivery AI Smart Matching System
### AI 기반 탁송/대리 기사 매칭 서비스 — 차량 흠집 탐지 · 감정 분석 · 최적 경로 추천

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=flat-square&logo=flask&logoColor=white)
![MySQL](https://img.shields.io/badge/MySQL-4479A1?style=flat-square&logo=mysql&logoColor=white)
![AWS](https://img.shields.io/badge/AWS_EC2-FF9900?style=flat-square&logo=amazonaws&logoColor=white)

<br>

탁송 차량과 대리 기사를 AI 기술로 매칭하는 통합 서비스입니다.  
기사 신뢰도 기반 자동 배차, 고객 리뷰 감정 분석, **탁송 전·후 차량 흠집 자동 탐지**, 최적 운행 경로 추천을 통합한 플랫폼입니다.

> 🗓 **개발 기간**: 2026.01.02 – 2026.01.16  
> 👥 **팀 구성**: 4인 팀 프로젝트

---

## 👥 팀 구성 & 역할 분담

| 팀원 | 담당 파트 | 주요 내용 |
|------|----------|----------|
| **강지수 ★** | **차량 흠집 탐지** | EfficientNet-B0 Transfer Learning, OpenCV 데이터 파이프라인, 심각도 라벨링 체계 설계, FastAPI 서버 구축 · AWS EC2 배포 |
| 장성우 | 매칭 알고리즘 | 기사 신뢰도 기반 자동 배차, 최적 운행 경로 추천 |
| 강대규 | 리뷰 감정 분석 | 고객 리뷰 키워드 AI 학습, 감정 분류 시스템 |
| 이상혁 | 서버 & DB | MySQL 설계, Flask 서버 구축, 웹 페이지 구현 |

---

## 📌 프로젝트 개요

차량 탁송 서비스에서는 아래와 같은 문제가 발생합니다.

- 기사와 화물 매칭이 전화·문자 등 **수작업 중심**으로 이루어짐
- 탁송 중 발생한 **차량 손상 책임 분쟁**
- 배차 지연, 경로 비효율, 수익 불투명

본 프로젝트는 이를 해결하기 위해 **AI와 데이터를 기반으로 탁송 매칭 과정을 자동화**했습니다.

---

## 🧠 주요 기능

---

### 🚗 차량 흠집 탐지 — 강지수

> 팀 프로젝트 이후 **EfficientNet-B0 고도화 · FastAPI 서버 · AWS EC2 배포**까지 독립 서비스로 발전시켰습니다.  
> [![GitHub](https://img.shields.io/badge/GitHub-vehicle--damage--detection-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/rkdwltn1211/vehicle-damage-detection)

탁송 전·후 차량 사진을 비교하여 신규 흠집 여부와 손상 심각도(0~4단계)를 자동 판정합니다.

- **EfficientNet-B0** Transfer Learning으로 심각도 분류 (val accuracy 64%)
- **OpenCV 파이프라인**: ECC 정렬 → Canny 엣지 → Morphology → ROI 패치 추출
- 소규모 데이터(126쌍) 한계를 데이터 증강으로 극복 (634개 → 4,190장)
- 클래스 불균형 대응: CrossEntropyLoss 가중치 · Early Stopping 적용
- 심각도 0~4단계 라벨링 기준 직접 설계

| 단계 | 수준 | 기준 |
|:---:|------|------|
| 0 | 정상 | 신규 흠집 없음 |
| 1 | 경미 | 얕은 스크래치, 멀리서 잘 안 보임 |
| 2 | 보통 | 육안으로 명확, 도장 손상 |
| 3 | 심각 | 넓거나 깊음, 찌그러짐, 수리 필요 |
| 4 | 매우 심각 | 구조적 손상, 즉시 수리·교체 필요 |

---

### 🧭 기사 매칭 시스템 — 장성우

주문 위치와 기사 위치 데이터를 기반으로 최적 기사를 자동 매칭합니다.

- 거리·시간·수익성을 고려한 스코어링 알고리즘 설계
- 조건 필터링 후 우선순위 기반 자동 배차
- 최적 운행 경로 추천 및 지도 화면 표시

---

### 💬 리뷰 감정 분석 — 강대규

고객 리뷰 텍스트를 분석하여 긍정/부정 감정을 분류하고 기사 신뢰도 점수에 반영합니다.

- 리뷰 텍스트 형태소 분석 및 정수 인코딩
- AI 모델 학습을 통한 긍정/부정 분류
- 신뢰 확률 기반 기사 점수 가산·감산

---

### 🗄️ 서버 & DB — 이상혁

서비스 전체의 백엔드 인프라와 데이터베이스를 구축했습니다.

- MySQL 스키마 설계 및 데이터 관리
- Flask 기반 REST API 서버 구축
- 웹 페이지 구현 및 프론트엔드 연동

---

## 📂 프로젝트 구조

```
vehicle-delivery-ai-system/
├── damage_detection/          # 🚗 차량 흠집 탐지 (강지수)
│   ├── notebooks/
│   │   ├── severity.ipynb     # CNN 모델 학습 & 추론
│   │   └── damege_box.ipynb   # OpenCV 데이터 파이프라인
│   └── model/checkpoints/
├── driver_matching/           # 🧭 기사 매칭 / 경로 최적화 (장성우)
│   └── notebooks/
├── review_sentiment/          # 💬 리뷰 감정 분석 (강대규)
│   └── notebooks/
├── database/mysql/            # 🗄️ DB 스키마 (이상혁)
└── README.md
```

---

## 🛠 Tech Stack

| 분류 | 기술 |
|------|------|
| 언어 | Python |
| 딥러닝 | PyTorch, EfficientNet-B0 |
| 이미지 처리 | OpenCV, PIL |
| 백엔드 | FastAPI (흠집 탐지), Flask (서버) |
| 배포 | AWS EC2 t3.micro (흠집 탐지 파트) |
| DB | MySQL |
| 기타 | Jupyter Notebook, Pandas, NumPy |

---

## ⚠️ 한계 및 개선 방향

| 한계 | 개선 방향 |
|------|----------|
| 126쌍 소규모 데이터 — val accuracy 64% | CarDD 등 공개 데이터셋 추가 결합 |
| Score 3·4 케이스 부족으로 F1 낮음 | 추가 수집 및 Focal Loss 적용 |
| 흠집 위치 시각화 미구현 | OpenCV 바운딩 박스 파이프라인 통합 예정 |

---

## 📝 One-line Summary

> **탁송 차량의 전·후 이미지를 EfficientNet-B0 CNN + OpenCV 파이프라인으로 분석하여 흠집 심각도(0~4)를 자동 판정하고, 기사 매칭·리뷰 감정 분석을 통합한 AI 기반 탁송 서비스를 구현**한 팀 프로젝트입니다.

---

## 👤 개발자

**강지수 (Kang Ji Soo)**  
📧 rkdwl3264@naver.com  
🐙 [github.com/rkdwltn1211](https://github.com/rkdwltn1211)
