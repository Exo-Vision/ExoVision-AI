데이터셋
- kepler, k2, tess 데이터셋 사용

문제점
CANDIDATE 라벨의 문제로 인해 모델의 성능이 하락됨
- 후보(Candidate)는 최종 상태가 정해지지 않은 중간 라벨로, 동일 표본이 추후 확인된 외계행성(Confirmed) 혹은 오탐(False Positive) 로 전이될 수 있습니다.
따라서 후보 라벨은 통계적으로 두 분포(행성/오탐)의 혼합(mixture) 이며, 개별 샘플의 참 라벨은 잠재변수로 숨겨져 있습니다.

데이터셋 문제점
1. kepler,k2 데이터셋과 tess 데이터셋의 칼럼명과 단위, 측정 방식의 극심한 차이
2. 데이터셋에 존재하는 많은 결측값

최종 데이터셋 선별 이유
현재 tess 프로젝트에서 tess dataset이 계속 추가가 되는 상황
하지만 tess 데이터셋에 존재하는 칼럼값으로는 후보군까지 판단 가능하며, 이후 추가 검증을 통해 외계행성임을 확정지어야함
추가 검증은 추가 데이터로 인한 진단 또는 ExoFOP/TFOP 후속관측이 필요함
하지만 kepler dataset은 외계행성임을 판단하기에 충분한 칼럼을 가지고 있음.
kepler 프로젝트 이후 진행한 k2 데이터셋은 kepler 프로젝트와 비슷한 부분이 많으므로 
kepler 데이터셋을 메인으로 k2, tess 데이터셋을 가공하여 kepler dataset에 합치는걸로 진행함
+
- 가장 많은 데이터 (전체의 45%)
- 핵심 피처 96% 이상 (결측치 적음)
- TESS는 핵심 feature 없음
- 데이터가 13년간 검증 완료
- TESS는 현재 진행중 (Candidate가 많음)
- kepler 데이터셋을 사용하는 자료가 많음

모델 성능 향상 노력
tess 프로젝트만 현재 외계행성을 관측하고 있기 때문에, tess 데이터셋에서 외계행성을 판별하는 주요한 칼럼들을 kepler dataset에 추가했다.

결측치 처리
물리 법칙 기반 계산
- 케플러 제3법칙을 이용한 반장축 계산
- 뉴턴의 중력 법칙을 이용하여 표면중력 계산
- 이를 포함한 7개 칼럼, 1만 3천여개 결측치 해결
통계적 추정
- 주기 기반 조건부 추정하여 궤도 이심률 계산
- 이를 포함한 4개 칼럼, 3만 3천여개 결측치 해결

모델
tabular dataset에 강점인 gradient boosting 계열 ML 모델 사용
xgboost, lightgbm, catboost, gradient boosting, extra trees, random forest 모델 시도
초기 시도 ensemble 모델 92% 정확도 획득
이후 외계행성, 비외계행성, 후보 3-class 예측 모델에서 74% 정확도 획득

최종 모델 아키텍쳐
```
입력 → 모델 1 (CONFIRMED vs FALSE POSITIVE)
     ↓
   확신도 체크
     ↓
  확신도 ≥ 0.96?
     ↓
  Yes → 모델 1 결과 출력
     ↓
  No → 모델 2 (IS_CANDIDATE?)
     ↓
     Yes → CANDIDATE
     ↓
     No → 모델 1 결과 출력
```

**모델 1: CONFIRMED vs FALSE POSITIVE**
- **알고리즘**: CatBoost (최고 성능)
- **정확도**: **89.49%**
- **과적합**: 1.04%p
- **CV**: 89.51% ± 0.55%

**모델 2: CANDIDATE 판별**
- **알고리즘**: Voting Ensemble (CatBoost + XGBoost + LightGBM)
- **정확도**: **74.39%**
- **과적합**: 1.98%p

**통합 시스템**:
- **최종 정확도**: **70.86%**
- **최적 임계값**: 0.96
- **1단계 사용률**: 22.8%


향후 개선 방향
단기 개선
**1. 추가 피처 엔지니어링**
```python
# 아직 시도하지 않은 피처들
df['habitable_zone_score'] = 1 / (1 + np.abs(df['koi_teq'] - 288))  # 지구 온도 = 288K
df['roche_limit'] = 2.46 * df['koi_srad'] * (df['koi_smass'] / df['koi_prad'])**(1/3)
df['hill_sphere'] = df['koi_sma'] * (df['koi_smass'] / 3)**(1/3)
```

**2. 하이퍼파라미터 세밀 튜닝**
- Grid Search / Bayesian Optimization
- CatBoost depth: 4 → 5~6 시도
- Learning rate: 0.02 → 0.01~0.03 범위

**3. 데이터 증강**
- SMOTE로 클래스 균형 맞추기
- 특히 CANDIDATE 클래스 증강

중기 개선
**1. 딥러닝 모델 도입**
```python
# 1D CNN for time-series (Light Curve)
# Transformer for attention mechanism
# Graph Neural Network for stellar-planetary system
```

**2. 앙상블 다양화**
- Neural Network + Tree-based 혼합
- Stacking의 Meta-learner 개선

**3. 결측치 처리 고도화**
- MICE (Multivariate Imputation by Chained Equations)
- KNN Imputer
- Deep Learning Imputer

장기 개선
**1. 추가 데이터 수집**
- Light Curve 원시 데이터
- 분광 데이터 (Spectroscopy)
- 후속 관측 데이터

**2. Transfer Learning**
- 사전 학습된 천문학 모델 활용
- Domain Adaptation

**3. Active Learning**
- 불확실성 높은 샘플 우선 라벨링
- 관측 자원 최적 배분

결론
#### 주요 성과
1. ✅ **세 망원경 데이터 성공적 통합**: 21,271개 샘플, 25개 피처
2. ✅ **결측치 77.8% 감소**: 물리 법칙 + ML + 통계
3. ✅ **이진 분류 89.49%**: CONFIRMED vs FALSE POSITIVE
4. ✅ **과적합 1.04%p**: 강한 정규화로 일반화 능력 확보
5. ✅ **2-모델 시스템 구축**: 3-클래스 70.86% (CANDIDATE 포함)

#### 목표 대비
- **목표**: 95% 정확도
- **달성**: 89.49% (이진), 70.86% (3-클래스)
- **갭**: 5.51%p (이진), 24.14%p (3-클래스)

#### 핵심 인사이트
1. **행성 반지름**이 가장 중요한 판별 지표 (13.68%)
2. **관측 편향**(적위)이 신호 품질에 영향 (8.06%)
3. **금속성**-거대행성 상관관계 확인 (6.26%)
4. **CANDIDATE는 본질적으로 판별 어려움** (추가 관측 필요)
5. **과적합 방지** > 복잡한 모델

#### 미래 전망
- 추가 피처 엔지니어링으로 **90~92% 도달 가능**
- 딥러닝 + Light Curve 원시 데이터로 **93~95% 잠재력**
- CANDIDATE 정확도 향상은 **추가 관측 데이터 필수**

#### 최종 평가
이 프로젝트는 **천체물리학 + 데이터 과학의 성공적 융합**을 보여줍니다. 비록 95% 목표에는 미달했지만, 89.49%의 이진 분류 정확도와 과적합 1.04%p는 **실제 운영 가능한 수준**입니다. 더 중요한 것은, 데이터 통합부터 모델 개발까지 **체계적이고 과학적인 접근법**을 확립했다는 점입니다.