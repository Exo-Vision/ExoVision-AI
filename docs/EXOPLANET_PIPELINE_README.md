# 외계행성 분류 파이프라인

NASA의 TESS, Kepler, K2 데이터를 사용한 외계행성 후보 분류 시스템

## 📋 프로젝트 개요

이 프로젝트는 두 가지 주요 분류 작업을 수행합니다:

1. **TESS 모델**: 외계행성 후보 vs 비후보 분류
   - TESS 데이터를 Kepler/K2 데이터로 증강
   - XGBoost를 사용한 이진 분류

2. **Kepler+K2 모델**: 외계행성 후보 vs 확정된 외계행성 분류
   - Kepler와 K2 데이터를 통합
   - 후보를 확정 행성으로 분류

## 🎯 주요 기능

### 1. TESS 데이터 유용한 컬럼 선택
- 외계행성 분류에 필요한 핵심 특성만 추출
- 행성 특성: 궤도 주기, 반지름, 통과 깊이, 평형 온도 등
- 별 특성: 온도, 반지름, 거리, 표면 중력 등

### 2. Kepler/K2 → TESS 형식 매핑
- Kepler와 K2의 컬럼을 TESS 형식으로 변환
- 라벨 통일: CONFIRMED → CP, CANDIDATE → PC, FALSE POSITIVE → FP

### 3. TESS 데이터 증강
- Kepler와 K2의 매핑된 데이터를 TESS에 추가
- 더 많은 학습 데이터 확보

### 4. XGBoost 모델 학습
- TESS: 외계행성 후보 여부 분류 (이진 분류)
- Kepler+K2: 후보 중 확정 행성 분류 (이진 분류)

### 5. Optuna 하이퍼파라미터 최적화
- 자동으로 최적의 모델 파라미터 탐색
- Cross-validation으로 과적합 방지

### 6. 모델 저장/로드
- 학습된 모델을 pickle 형식으로 저장
- 최적 파라미터를 JSON 형식으로 저장
- 언제든지 모델 재사용 가능

## 📁 프로젝트 구조

```
nasa/
├── datasets/                          # 데이터셋 디렉토리
│   ├── tess.csv                      # 원본 TESS 데이터
│   ├── kepler.csv                    # 원본 Kepler 데이터
│   ├── k2.csv                        # 원본 K2 데이터
│   ├── tess_useful_columns.csv       # 선택된 TESS 컬럼
│   ├── kepler_mapped_to_tess.csv     # TESS 형식으로 변환된 Kepler
│   ├── k2_mapped_to_tess.csv         # TESS 형식으로 변환된 K2
│   ├── tess_augmented.csv            # 증강된 TESS 데이터
│   └── kepler_k2_merged.csv          # Kepler+K2 통합 데이터
├── models/                            # 모델 저장 디렉토리
│   ├── tess_model_*.pkl              # TESS 분류 모델
│   ├── tess_params_*.json            # TESS 모델 파라미터
│   ├── kepler_k2_model_*.pkl         # Kepler+K2 분류 모델
│   └── kepler_k2_params_*.json       # Kepler+K2 모델 파라미터
└── exoplanet_classification_pipeline.py  # 메인 파이프라인
```

## 🚀 사용 방법

### 기본 실행 (최적화 없이)

```python
from exoplanet_classification_pipeline import ExoplanetClassificationPipeline

# 파이프라인 생성
pipeline = ExoplanetClassificationPipeline(
    data_dir='datasets',
    models_dir='models'
)

# 전체 파이프라인 실행 (기본 파라미터 사용)
results = pipeline.run_full_pipeline(use_optuna=False)
```

### Optuna 최적화 포함 실행

```python
# Optuna로 하이퍼파라미터 최적화 (권장)
results = pipeline.run_full_pipeline(
    use_optuna=True,  # Optuna 사용
    n_trials=50       # 50회 시도 (더 많을수록 좋지만 시간 소요)
)
```

### 개별 단계 실행

```python
# 1. TESS 유용한 컬럼 선택
tess_filtered, tess_columns = pipeline.select_useful_tess_columns()

# 2. Kepler/K2를 TESS 형식으로 매핑
kepler_mapped, k2_mapped = pipeline.map_kepler_k2_to_tess()

# 3. TESS 데이터 증강
augmented_tess = pipeline.augment_tess_data(
    tess_filtered, kepler_mapped, k2_mapped
)

# 4. TESS 모델 학습
X_tess, y_tess, features = pipeline.prepare_tess_data_for_training(augmented_tess)
tess_model, tess_data = pipeline.train_tess_model(
    X_tess, y_tess, 
    use_optuna=True, 
    n_trials=30
)

# 5. Kepler + K2 통합
merged_data = pipeline.merge_kepler_k2()

# 6. Kepler+K2 모델 학습
X_kepler, y_kepler, features = pipeline.prepare_kepler_k2_data_for_training(merged_data)
kepler_model, kepler_data = pipeline.train_kepler_k2_model(
    X_kepler, y_kepler,
    use_optuna=True,
    n_trials=30
)

# 7. 모델 저장
pipeline.save_models(prefix='optimized_')
```

### 저장된 모델 로드

```python
# 모델 로드
pipeline.load_models(
    tess_model_path='models/tess_model_20241005_120000.pkl',
    kepler_k2_model_path='models/kepler_k2_model_20241005_120000.pkl'
)

# 예측
predictions = pipeline.tess_model.predict(X_new_data)
```

## 📊 데이터셋 설명

### TESS (Transiting Exoplanet Survey Satellite)
- **목적**: 외계행성 후보 vs 비후보 분류
- **레이블**: 
  - PC (Planetary Candidate) → 후보 (1)
  - CP (Confirmed Planet) → 후보 (1)
  - FP (False Positive) → 비후보 (0)
  - FA (False Alarm) → 비후보 (0)

### Kepler & K2
- **목적**: 외계행성 후보 vs 확정된 외계행성 분류
- **레이블**:
  - CANDIDATE → 후보 (0)
  - CONFIRMED → 확정 (1)

## 🔧 주요 파라미터

### TESS 유용한 컬럼 (선택된 특성)

**행성 특성:**
- `pl_orbper`: 궤도 주기 (days)
- `pl_trandurh`: 통과 지속시간 (hours)
- `pl_trandep`: 통과 깊이 (ppm)
- `pl_rade`: 행성 반지름 (지구 반지름)
- `pl_insol`: 일조량 (지구 대비)
- `pl_eqt`: 평형 온도 (K)

**별 특성:**
- `st_tmag`: TESS magnitude
- `st_dist`: 거리 (parsec)
- `st_teff`: 별 온도 (K)
- `st_logg`: 표면 중력
- `st_rad`: 별 반지름 (태양 반지름)

### XGBoost 하이퍼파라미터 (Optuna 탐색 범위)

- `max_depth`: 3~10 (트리 깊이)
- `learning_rate`: 0.01~0.3
- `n_estimators`: 50~300 (트리 개수)
- `min_child_weight`: 1~10
- `subsample`: 0.6~1.0
- `colsample_bytree`: 0.6~1.0
- `gamma`: 0~5
- `reg_alpha`: 0~10 (L1 정규화)
- `reg_lambda`: 0~10 (L2 정규화)

## 📈 성능 평가 지표

모델 학습 후 다음 지표들이 출력됩니다:

1. **Accuracy**: 전체 정확도
2. **F1 Score**: 정밀도와 재현율의 조화 평균
3. **Classification Report**: 클래스별 성능
   - Precision (정밀도)
   - Recall (재현율)
   - F1-score
4. **Confusion Matrix**: 혼동 행렬
5. **Feature Importance**: 특성 중요도 Top 10

## 💡 팁 & 주의사항

### 성능 향상 팁
1. **Optuna 사용 권장**: `use_optuna=True`로 설정하면 최적 파라미터 자동 탐색
2. **충분한 trials 설정**: `n_trials=50~100` 권장 (시간이 허락하는 범위에서)
3. **데이터 전처리**: 결측치와 이상치는 자동으로 처리되지만, 필요시 추가 전처리 가능

### 주의사항
1. **메모리**: 큰 데이터셋의 경우 충분한 RAM 필요
2. **시간**: Optuna 최적화는 시간이 오래 걸릴 수 있음 (n_trials에 비례)
3. **클래스 불균형**: 데이터셋의 클래스 불균형이 심한 경우 성능 저하 가능

## 🔍 결과 해석

### TESS 모델 결과
- **목적**: 새로운 관측 대상이 외계행성 후보인지 판별
- **활용**: 관측 우선순위 결정, 후속 관측 계획 수립

### Kepler+K2 모델 결과
- **목적**: 후보 중 실제 외계행성일 가능성 평가
- **활용**: 확정 발표 전 검증, 후속 연구 대상 선정

## 📚 참고 자료

- [TESS 데이터 컬럼 설명](https://exoplanetarchive.ipac.caltech.edu/docs/API_TOI_columns.html)
- [Kepler 데이터 컬럼 설명](https://exoplanetarchive.ipac.caltech.edu/docs/API_kepcandidate_columns.html)
- [K2 데이터 컬럼 설명](https://exoplanetarchive.ipac.caltech.edu/docs/API_k2pandc_columns.html)

## 🛠️ 필수 라이브러리

```bash
pip install pandas numpy scikit-learn xgboost optuna
```

## 📝 라이선스

이 프로젝트는 NASA 공개 데이터를 사용합니다.

## 👥 기여

문제 발견 시 이슈를 생성하거나 풀 리퀘스트를 제출해주세요.

---

**행복한 외계행성 탐색 되세요! 🌟🪐**
