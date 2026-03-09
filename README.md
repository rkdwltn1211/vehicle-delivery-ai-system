# 🚚 Vehicle Delivery AI System

차량 탁송 서비스에서 활용할 수 있는 **AI 기반 차량 관리 및 매칭 시스템**입니다.
차량 상태 분석, 기사 매칭 최적화, 고객 리뷰 분석을 통해 탁송 서비스의 효율성과 품질을 개선하는 것을 목표로 합니다.

---

## 📌 프로젝트 개요

차량 탁송 서비스에서는 차량 상태 확인, 기사 매칭, 고객 서비스 평가 등 여러 요소가 중요합니다.
본 프로젝트는 이러한 문제를 해결하기 위해 **AI 기반 분석 시스템**을 구축했습니다.

주요 기능

* 차량 이미지 분석을 통한 **흠집 판정**
* 주문 위치 기반 **최적 기사 매칭**
* 고객 리뷰 기반 **감정 분석**

---

## 🧠 주요 기능

### 🚗 차량 흠집 판정 (Damage Detection)

탁송 전 / 후 차량 사진을 비교하여
**신규 흠집 여부 및 손상 정도**를 판정하는 모델입니다.

기능

* 차량 이미지 전처리
* CNN 기반 흠집 판정 모델
* 손상 정도 분류

---

### 🧭 기사 매칭 시스템 (Driver Matching)

주문 위치와 기사 위치 데이터를 기반으로
**가장 가까운 기사 및 최적 경로를 계산**하여 매칭합니다.

기능

* 기사 위치 기반 거리 계산
* 최적 기사 매칭 알고리즘
* 배송 경로 최적화

---

### 💬 리뷰 감정 분석 (Review Sentiment Analysis)

고객 리뷰 텍스트를 분석하여
**긍정 / 부정 감정**을 분류하는 모델입니다.

기능

* 텍스트 전처리
* 감정 분석 모델
* 리뷰 데이터 분석

---

## 👨‍💻 나의 역할 (Contribution)

본 프로젝트에서 다음 기능을 직접 구현했습니다.

* **차량 흠집 판정 모델 개발**
* 차량 이미지 전처리 및 데이터 분석
* CNN 기반 이미지 분류 모델 구현
* 모델 학습 및 성능 평가
* 프로젝트 폴더 구조 설계 및 GitHub 관리

---

## 📂 프로젝트 구조

```
vehicle-delivery-ai-system
│
├ damage_detection        # 차량 흠집 판정
│ ├ data
│ ├ notebooks
│ ├ model
│ └ result
│
├ driver_matching         # 기사 매칭 / 경로 최적화
│ ├ data
│ │ ├ delivery_orders_2026.csv
│ │ └ delivery_orders_2026_utf8_version.csv
│ │
│ ├ notebooks
│ │ └ driver_matching_model.ipynb
│ 
│ 
│
├ review_sentiment        # 리뷰 감정 분석
│ ├ data
│ │ ├ drivers_2026.xlsx
│ │ └ review_label.csv
│ │
│ ├ notebooks
│ │ └ review_sentiment_analysis.ipynb
│ 
│
├ database
│ └ mysql
│
├ docs
│ ├ 탁송흐름도.png
│ └ 시스템설명.pdf
│
└ README.md
```

---

## 🛠 사용 기술

Language

* Python

Data Analysis

* Pandas
* Numpy

Machine Learning

* Scikit-learn
* TensorFlow

Visualization

* Matplotlib
* Seaborn

Database

* MySQL

---

## 📊 기대 효과

* 차량 상태 자동 분석을 통한 **품질 관리 개선**
* 효율적인 기사 매칭으로 **배송 시간 단축**
* 고객 리뷰 분석을 통한 **서비스 품질 개선**

---


