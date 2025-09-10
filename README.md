# Dacon Basic 대회용 머신러닝 프로젝트

## 프로젝트 개요

이 프로젝트는 Freebies' MLDL Mini Project로 [Dacon 대회](https://dacon.io/competitions/official/236590/overview/description)를 대상으로 하며, **다중 분류** 문제를 다룹니다. 목표는 익명화된 블랙박스 특성(`X_01`, `X_02`, …)으로부터 정상/비정상 유형(0–20)을 예측하는 모델을 설계하는 것입니다.

- **문제 유형**: 다중 분류(21 클래스, 0–20)  
- **평가 지표**: Macro-F1 Score  
- **데이터 규모**: Train 21,693개, Test 15,004개

저장소에는 EDA 스크립트부터 고전 ML, GBDT, 그리고 딥러닝 기반 Transformer 모델까지 다양한 구현이 포함되어 있습니다.

## 저장소 구조

```
FISA_MachineLearning-prac/
├───.gitignore
├───README.md
├───requirements.txt
├───data/
│   ├───sample_submission.csv
│   ├───test.csv
│   └───train.csv
└───src/
    ├───eda.py
    ├───fttransformer.ipynb
    ├───knn.py
    ├───lightgbm.py
    ├───poly.py
    ├───tabnet.ipynb
    └───xgb.py
```

- **`data/`**: 학습 데이터(`train.csv`), 테스트 데이터(`test.csv`), 제출 예시 파일.  
- **`src/`**: 분석 및 모델링을 위한 파이썬 스크립트/노트북.

## 공통 전략

여러 모델에 공통으로 아래 전략을 적용해 **견고성**과 **성능**을 확보했습니다.

- **이상치 처리**: 트리 기반 모델 학습 안정화를 위해 0.5%와 99.5% **분위수 클리핑(winsorizing)** 적용.  
- **교차 검증**: **Stratified 5-Fold CV**로 데이터 활용 효율 및 일반화 성능 향상.  
- **앙상블**: GBDT/Transformer 계열은 **다른 랜덤 시드**로 재학습한 예측을 **평균(Out-of-Fold 기반)** 하여 최종 제출 생성.

## 모델 성능

| 모델 | Macro-F1 |
| :--- | :---: |
| KNN | 0.5569 |
| TabNet | 0.7007 |
| LightGBM | 0.7445 |
| XGBoost | 0.7594 |
| **FT-Transformer** | **0.8205** |

*점수는 프로젝트 발표 문서 기준입니다.*

## 환경 설정

1) **저장소 클론**
```bash
git clone <repository-url>
cd FISA_MachineLearning-prac
```

2) **가상환경 생성(권장)**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scriptsctivate
```

3) **필요 패키지 설치**
```bash
pip install -r requirements.txt
```

## 사용법

`src/`의 각 스크립트는 **독립 실행** 가능하며 학습 후 제출 파일을 생성합니다(프로젝트 루트에서 실행 가정).

```bash
# LightGBM 실행
python src/lightgbm.py

# XGBoost 실행
python src/xgb.py
```
생성된 제출 파일은 `data/` 디렉터리에 저장됩니다.

## 모델 설명

### 1. 탐색적 데이터 분석(EDA) — `eda.py`
- 정보량이 높고 중복이 적은 특성 선별.  
- 다중공선성(VIF), 상관관계, 분포 유사도(KL Divergence) 등을 점검하여 **23개 특성**을 선정.

### 2. K-Nearest Neighbors — `knn.py`
- 거리 기반 **비모수 분류기**.  
- EDA에서 고른 **23개 특성**만 사용.  
- **차원의 저주**와 **노이즈 민감성**으로 본 과제에선 성능이 낮게 나타남.

### 3. 다항 회귀 — `poly.py`
- 다항 특성으로 **비선형성**을 선형 모델로 포착.  
- 마찬가지로 선별된 특성 하위셋 사용.

### 4. TabNet — `tabnet.ipynb`
- 매 **의사결정 단계(decision step)**마다 **순차적 어텐션**으로 중요한 특성만 선택하여 추론하는 **딥러닝 탭룰라 모델**.  
- **구현**: 학습/검증 분할과 **커스텀 Macro-F1**로 평가.

### 5. LightGBM — `lightgbm.py`
- **leaf-wise** 트리 성장의 고성능 GBDT 프레임워크.  
- **구현**: GPU(`device: 'gpu'`), Stratified 5-Fold CV, **3개 시드 평균**.  
- 출력: `data/lgbm_oof_seedavg_submit.csv`.

### 6. XGBoost — `xgb.py`
- 규제가 강하고 안정적인 **GBDT 정석 구현**.  
- **구현**: GPU(`tree_method: 'gpu_hist'`), Stratified 5-Fold CV, **3개 시드 평균**.  
- 출력: `data/xgb_oof_seedavg_submit.csv`.

### 7. FT-Transformer — `fttransformer.ipynb`
- 탭룰라에 **Transformer self-attention**을 적용.  
- **핵심 아이디어**: **특성마다 토큰화**하여 임베딩, **CLS 토큰**으로 샘플 표현 학습 → 복잡한 특성 상호작용을 자동 학습.  
- **구현**: Stratified 5-Fold, AdamW, **ReduceLROnPlateau + Early Stopping**으로 과적합 억제.  
- 출력: `data/fttransformer_5fold_submit.csv`.
