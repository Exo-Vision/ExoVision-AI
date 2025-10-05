# 🌟 외계행성 예측 가이드

## 📋 모델 필수 입력값

### TESS 모델 (외계행성 후보 판별) - 16개 입력값

| 번호 | 컬럼명        | 설명           | 단위       | 예시 값 |
| ---- | ------------- | -------------- | ---------- | ------- |
| 1    | `pl_orbper`   | 궤도 주기      | days       | 10.5    |
| 2    | `pl_trandurh` | 통과 지속시간  | hours      | 3.2     |
| 3    | `pl_trandep`  | 통과 깊이      | ppm        | 1500    |
| 4    | `pl_rade`     | 행성 반지름    | 지구=1     | 1.2     |
| 5    | `pl_insol`    | 일조량         | 지구=1     | 50      |
| 6    | `pl_eqt`      | 평형 온도      | Kelvin     | 500     |
| 7    | `pl_tranmid`  | 통과 중간 시간 | BJD        | 2459000 |
| 8    | `st_tmag`     | TESS magnitude | mag        | 12.5    |
| 9    | `st_dist`     | 거리           | parsec     | 100     |
| 10   | `st_teff`     | 별 온도        | Kelvin     | 5800    |
| 11   | `st_logg`     | 별 표면 중력   | log(cm/s²) | 4.5     |
| 12   | `st_rad`      | 별 반지름      | 태양=1     | 1.0     |
| 13   | `st_pmra`     | 적경 고유운동  | mas/yr     | -5.0    |
| 14   | `st_pmdec`    | 적위 고유운동  | mas/yr     | -3.0    |
| 15   | `ra`          | 적경           | degrees    | 123.456 |
| 16   | `dec`         | 적위           | degrees    | 45.678  |

**예측 결과:**
- `0`: ❌ 비후보 (외계행성이 아님)
- `1`: ✅ 후보 (외계행성 가능성 높음)

---

## 🚀 빠른 시작

### 1. 단일 데이터 예측

```python
from predict_exoplanet import ExoplanetPredictor

# 예측기 생성 및 모델 로드
predictor = ExoplanetPredictor()
predictor.load_latest_models()

# 관측 데이터
observation = {
    'pl_orbper': 10.5,
    'pl_trandurh': 3.2,
    'pl_trandep': 1500,
    'pl_rade': 1.2,
    'pl_insol': 50,
    'pl_eqt': 500,
    'pl_tranmid': 2459000,
    'st_tmag': 12.5,
    'st_dist': 100,
    'st_teff': 5800,
    'st_logg': 4.5,
    'st_rad': 1.0,
    'st_pmra': -5.0,
    'st_pmdec': -3.0,
    'ra': 123.456,
    'dec': 45.678
}

# 예측
prediction, probability = predictor.predict_tess(observation)

print(f"예측: {'외계행성 후보' if prediction[0] == 1 else '비후보'}")
print(f"확률: {probability[0][1]:.2%}")
```

### 2. CSV 파일로 배치 예측

```python
import pandas as pd
from predict_exoplanet import ExoplanetPredictor

# 예측기 로드
predictor = ExoplanetPredictor()
predictor.load_latest_models()

# CSV 파일 읽기
data = pd.read_csv('my_observations.csv')

# 배치 예측
predictions, probabilities = predictor.predict_tess(data)

# 결과 저장
results = pd.DataFrame({
    'prediction': predictions,
    'candidate_prob': probabilities[:, 1]
})
results.to_csv('predictions.csv', index=False)
```

### 3. 샘플 입력 파일 생성

```bash
# 터미널에서 실행
conda run -n nasa python predict_exoplanet.py
```

이 명령어는 다음을 수행합니다:
1. 필수 입력 변수 목록 출력
2. 각 변수의 상세 설명 출력
3. 모델 로드
4. 예제 실행
5. `sample_input.csv` 생성

생성된 `sample_input.csv`를 수정하여 사용하세요!

---

## 📊 입력 데이터 형식

### Option 1: 딕셔너리 (단일 예측)

```python
data = {
    'pl_orbper': 10.5,
    'pl_trandurh': 3.2,
    # ... 나머지 14개 컬럼
}
```

### Option 2: 리스트 (여러 개 예측)

```python
data = [
    {'pl_orbper': 10.5, 'pl_trandurh': 3.2, ...},
    {'pl_orbper': 5.2, 'pl_trandurh': 2.1, ...},
]
```

### Option 3: DataFrame (CSV에서 읽기)

```python
data = pd.read_csv('observations.csv')
```

**CSV 형식 예시:**
```csv
pl_orbper,pl_trandurh,pl_trandep,pl_rade,pl_insol,pl_eqt,pl_tranmid,st_tmag,st_dist,st_teff,st_logg,st_rad,st_pmra,st_pmdec,ra,dec
10.5,3.2,1500,1.2,50,500,2459000,12.5,100,5800,4.5,1.0,-5.0,-3.0,123.456,45.678
5.2,2.1,800,0.9,100,600,2459100,11.2,80,6000,4.3,1.1,-3.0,-2.0,200.123,-30.456
```

---

## 💡 주요 변수 설명

### 행성 특성

- **pl_orbper** (궤도 주기): 행성이 별 주위를 한 바퀴 도는 시간
  - 지구: 365일
  - 수성: 88일
  - 목성: 4333일

- **pl_trandurh** (통과 시간): 행성이 별 앞을 지나가는 시간
  - 짧을수록 작은 행성 or 빠른 궤도
  - 보통 2~8시간

- **pl_trandep** (통과 깊이): 별 밝기가 얼마나 감소하는가
  - 클수록 큰 행성
  - 단위: ppm (parts per million)
  - 지구급 행성: ~100 ppm
  - 목성급 행성: ~10000 ppm

- **pl_rade** (행성 반지름): 지구를 1로 했을 때 크기
  - 0.5 = 지구의 절반 크기
  - 2.0 = 지구의 2배 크기
  - 11.0 = 목성 크기

- **pl_insol** (일조량): 행성이 받는 에너지
  - 1 = 지구가 받는 태양 에너지
  - > 1 = 뜨거운 행성
  - < 1 = 차가운 행성

### 별 특성

- **st_teff** (별 온도):
  - 태양: 5778 K
  - 차가운 별: 3000~4000 K (적색왜성)
  - 뜨거운 별: 7000~10000 K (청색왜성)

- **st_rad** (별 반지름):
  - 1.0 = 태양 크기
  - < 1 = 작은 별
  - > 1 = 큰 별

- **st_dist** (거리):
  - parsec 단위 (1 pc = 3.26 광년)
  - 가까운 별: 1~10 pc
  - 먼 별: 100~1000 pc

---

## 📈 결과 해석

### 확률 해석

```python
predictions, probabilities = predictor.predict_tess(data)

# probabilities 형식: [비후보 확률, 후보 확률]
non_candidate_prob = probabilities[0][0]  # 0~1
candidate_prob = probabilities[0][1]      # 0~1
```

**예시:**
- `[0.15, 0.85]`: 85% 확률로 외계행성 후보 → ✅ 강력한 후보
- `[0.45, 0.55]`: 55% 확률로 후보 → ⚠️ 불확실
- `[0.90, 0.10]`: 10% 확률로 후보 → ❌ 비후보 가능성 높음

### 권장 기준

- **확률 > 80%**: 매우 강력한 후보, 후속 관측 권장
- **확률 60~80%**: 유망한 후보, 추가 검증 필요
- **확률 40~60%**: 불확실, 더 많은 데이터 필요
- **확률 < 40%**: 비후보 가능성 높음

---

## 🔧 문제 해결

### 1. 모델을 찾을 수 없다는 오류

```
⚠️ TESS 모델을 찾을 수 없습니다.
```

**해결책:** 먼저 모델을 학습하세요
```bash
conda run -n nasa python run_pipeline_fast.py
```

### 2. 필수 컬럼 누락 오류

```
ValueError: 필수 컬럼 누락: {'pl_orbper', 'pl_rade'}
```

**해결책:** 누락된 컬럼을 추가하세요. 값이 없으면 0 또는 적절한 기본값 사용

```python
data['pl_orbper'] = 0  # 또는 적절한 값
```

### 3. 결측치가 많을 때

모델은 자동으로 결측치를 0으로 채우지만, 정확한 예측을 위해서는:
- 가능한 한 실제 값 사용
- 비슷한 천체의 평균값 사용
- 물리적으로 타당한 값 추정

---

## 📞 도움말

### 필수 입력값 확인

```python
predictor = ExoplanetPredictor()
predictor.show_required_features()
```

### 상세 설명 보기

```bash
conda run -n nasa python predict_exoplanet.py
```

### 예제 실행

```python
from predict_exoplanet import example_single_prediction

example_single_prediction()
```

---

## ⚡ 성능

**TESS 모델:**
- 정확도: 87.16%
- F1 Score: 91.02%
- 후보 탐지율: 94%

**처리 속도:**
- 단일 예측: < 0.01초
- 1000개 배치: < 0.1초

---

## 🎯 실전 활용 예시

### 케이스 1: 새로운 TESS 관측 데이터

```python
# 관측 데이터
new_observation = {
    'pl_orbper': 3.5,      # 3.5일 주기
    'pl_trandurh': 2.8,    # 2.8시간 통과
    'pl_trandep': 2500,    # 큰 행성
    'pl_rade': 8.5,        # 목성급
    'st_teff': 5200,       # K형 별
    # ... 나머지
}

pred, prob = predictor.predict_tess(new_observation)

if pred[0] == 1 and prob[0][1] > 0.8:
    print("🎉 강력한 외계행성 후보 발견!")
    print("후속 관측을 권장합니다.")
```

### 케이스 2: 대량 후보 필터링

```python
# 1000개 후보 데이터
candidates = pd.read_csv('tess_candidates.csv')

# 예측
preds, probs = predictor.predict_tess(candidates)

# 높은 확률만 필터링
high_confidence = candidates[probs[:, 1] > 0.8]

print(f"1000개 중 {len(high_confidence)}개가 강력한 후보입니다.")
high_confidence.to_csv('high_confidence_candidates.csv')
```

---

**Happy Planet Hunting! 🌟🪐**
