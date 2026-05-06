# Exoplanet Classification Model

NASA Space Apps Challenge 2025에서 진행한 외계행성 후보 판별 프로젝트입니다.  
Kepler, K2, TESS에서 제공하는 관측 데이터를 통합하고, 외계행성 후보를 `CONFIRMED`, `FALSE POSITIVE`, `CANDIDATE`로 분류하는 머신러닝 모델을 구축했습니다.

## Project Overview

본 프로젝트의 목표는 다양한 우주 관측 데이터셋을 하나의 학습 가능한 형태로 통합하고, 외계행성 후보의 물리적 특성을 기반으로 실제 행성 여부를 판별하는 것입니다.

단순히 하나의 3-class 분류 모델을 학습하는 방식이 아니라,  
`CONFIRMED`와 `FALSE POSITIVE`를 먼저 구분한 뒤, 확신도가 낮은 샘플에 대해 `CANDIDATE` 여부를 추가로 판별하는 계층적 분류 구조를 설계했습니다.

## Dataset

본 프로젝트에서는 Kepler, K2, TESS 세 가지 외계행성 관측 데이터셋을 사용했습니다.  
각 데이터셋은 컬럼명, 단위, 결측치 구조가 서로 다르기 때문에, 공통 컬럼 체계로 변환한 뒤 하나의 통합 데이터셋으로 구성했습니다.

최종 통합 데이터셋은 다음과 같습니다.

- Rows: 21,271
- Columns: 25
- Data Sources:
  - Kepler: 9,564 samples
  - TESS: 7,703 samples
  - K2: 4,004 samples
- Labels:
  - CANDIDATE: 8,592
  - FALSE POSITIVE: 6,329
  - CONFIRMED: 6,328
  - REFUTED: 22

주요 피처로는 궤도 주기, 행성 반지름, 항성 반지름, 항성 질량, 평형 온도, 통과 깊이, 통과 지속시간 등이 포함됩니다.

## Data Preprocessing

데이터 전처리는 크게 세 단계로 진행했습니다.

### 1. Dataset Integration

Kepler, K2, TESS 데이터셋을 공통 컬럼 기준으로 정리하고 하나의 데이터셋으로 통합했습니다.  
데이터셋별로 서로 다른 컬럼명을 `koi_period`, `koi_prad`, `koi_srad`, `koi_smass` 등 공통 피처명으로 매핑했으며, 데이터 출처를 구분하기 위해 `data_source` 컬럼을 추가했습니다.

또한 일부 데이터셋에 존재하지 않는 항성 질량, 궤도 반장축, 궤도 경사각 등은 물리 공식 기반으로 보완했습니다.

### 2. Missing Value Handling

결측치는 단순 평균 대체가 아니라, 변수의 물리적 의미에 따라 처리했습니다.

- 항성 질량: 항성 반지름과 표면중력 기반 계산
- 궤도 반장축: 케플러 제3법칙 기반 계산
- 표면중력: 항성 질량과 반지름 기반 계산
- 충격 매개변수: 반장축, 항성 반지름, 궤도 경사각 기반 계산
- 통과 깊이: 행성 반지름과 항성 반지름 기반 계산
- 입사 플럭스 및 평형 온도: 항성 온도, 반지름, 궤도 거리 기반 계산

물리 공식으로 보완하기 어려운 일부 컬럼은 Random Forest Regressor를 활용해 회귀 기반으로 결측치를 추정했습니다.

### 3. Error Column Processing

각 관측값에 포함된 상·하한 오차 컬럼은 변수별 특성에 맞게 하나의 error feature로 통합했습니다.

- 일반적인 관측 오차는 상·하한 절댓값의 평균 사용
- 보수적으로 처리해야 하는 항성 온도, 표면중력, 항성 질량 등은 최대 오차 사용
- 이심률처럼 비대칭성이 중요한 변수는 상한/하한 오차를 분리 보존
- K2, TESS의 limit flag는 필요한 경우 별도 컬럼으로 유지

이를 통해 데이터셋별로 다른 오차 표현 방식을 일관된 형태로 정리했습니다.

## Feature Engineering

기본 수치형 피처 외에도 외계행성 판별에 유의미할 수 있는 파생 피처를 추가했습니다.

- `planet_star_ratio`: 행성 반지름 / 항성 반지름
- `orbital_energy`: 궤도 반장축의 역수
- `transit_signal`: 통과 깊이 × 통과 지속시간
- `stellar_density`: 항성 질량 / 항성 반지름³
- `planet_density_proxy`: 행성 반지름과 궤도 거리 기반 밀도 proxy
- `log_period`: 궤도 주기의 로그 변환
- `log_depth`: 통과 깊이의 로그 변환
- `log_insol`: 입사 플럭스의 로그 변환
- `orbit_stability`: 이심률 × 충격 매개변수
- `transit_snr`: 통과 깊이 / 통과 지속시간

최종적으로 29개의 피처를 사용해 모델을 학습했습니다.

## Modeling Approach

본 프로젝트에서는 단일 3-class 분류 모델 대신, 두 개의 binary classifier를 조합한 계층적 분류 시스템을 사용했습니다.

### Model 1: Confirmed vs False Positive

첫 번째 모델은 이미 판정이 명확한 샘플을 대상으로 `CONFIRMED`와 `FALSE POSITIVE`를 구분합니다.  
CatBoost, XGBoost, LightGBM, Voting Ensemble을 비교하여 가장 높은 성능을 보이는 모델을 선택했습니다.

### Model 2: Candidate Detection

두 번째 모델은 전체 데이터셋을 대상으로 해당 샘플이 `CANDIDATE`인지 아닌지를 판별합니다.  
CatBoost, XGBoost, LightGBM, Neural Network, Voting Ensemble을 비교했습니다.

### Confidence-Based Pipeline

최종 예측은 다음과 같은 방식으로 수행됩니다.

1. Model 1이 높은 확신도로 예측한 경우  
   → `CONFIRMED` 또는 `FALSE POSITIVE`로 분류

2. Model 1의 확신도가 낮은 경우  
   → Model 2를 사용해 `CANDIDATE` 여부를 추가 판별

3. 여러 confidence threshold를 실험하여 최종 3-class 정확도가 가장 높은 임계값을 선택

이 구조를 통해 `CONFIRMED`, `FALSE POSITIVE`, `CANDIDATE`를 한 번에 분류하는 방식보다 후보군 판별에 더 적합한 흐름을 만들고자 했습니다.

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- LightGBM
- CatBoost
- Joblib

## Key Contributions

- Kepler, K2, TESS 데이터셋 통합 파이프라인 구현
- 데이터셋별 컬럼명 및 단위 차이를 공통 스키마로 정규화
- 물리 법칙 기반 결측치 보완 로직 설계
- 회귀 모델을 활용한 주요 피처 결측치 추정
- 관측 오차 컬럼 통합 및 정제
- 외계행성 판별에 필요한 파생 피처 설계
- 확신도 기반 2-stage hierarchical classification pipeline 구현
- 모델, 스케일러, 설정값 저장 및 재사용 가능한 예측 코드 구성

## Result

최종 모델은 다음과 같은 계층적 구조로 외계행성 후보를 분류합니다.

- Model 1: `CONFIRMED` vs `FALSE POSITIVE`
- Model 2: `CANDIDATE` vs `NOT_CANDIDATE`
- Final Pipeline: confidence threshold 기반 3-class classification

이를 통해 단순 분류 정확도뿐 아니라, 실제 외계행성 후보 판별 과정에서 중요한 `CANDIDATE` 샘플을 별도로 고려할 수 있는 구조를 구현했습니다.
