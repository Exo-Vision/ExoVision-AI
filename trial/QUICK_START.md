# 🚀 외계행성 예측 - 빠른 시작 가이드

## 📊 모델 입력값 요약

### TESS 모델 (16개 입력값)

```python
observation = {
    # 🪐 행성 특성 (7개)
    'pl_orbper': 10.5,      # 궤도 주기 (일)
    'pl_trandurh': 3.2,     # 통과 시간 (시간)
    'pl_trandep': 1500,     # 통과 깊이 (ppm)
    'pl_rade': 1.2,         # 행성 반지름 (지구=1)
    'pl_insol': 50,         # 일조량 (지구=1)
    'pl_eqt': 500,          # 온도 (K)
    'pl_tranmid': 2459000,  # 관측 시간 (BJD)
    
    # ⭐ 별 특성 (7개)
    'st_tmag': 12.5,        # 밝기
    'st_dist': 100,         # 거리 (pc)
    'st_teff': 5800,        # 온도 (K)
    'st_logg': 4.5,         # 표면 중력
    'st_rad': 1.0,          # 반지름 (태양=1)
    'st_pmra': -5.0,        # 고유운동 RA
    'st_pmdec': -3.0,       # 고유운동 Dec
    
    # 📍 위치 (2개)
    'ra': 123.456,          # 적경
    'dec': 45.678           # 적위
}
```

## 💻 코드 사용법

### 방법 1: 단일 예측
```python
from predict_exoplanet import ExoplanetPredictor

predictor = ExoplanetPredictor()
predictor.load_latest_models()

# 예측
pred, prob = predictor.predict_tess(observation)

print(f"결과: {'외계행성 후보' if pred[0] == 1 else '비후보'}")
print(f"확률: {prob[0][1]:.2%}")
```

### 방법 2: CSV 파일 예측
```python
import pandas as pd

data = pd.read_csv('my_data.csv')
predictions, probabilities = predictor.predict_tess(data)
```

### 방법 3: 여러 개 예측
```python
observations = [
    {'pl_orbper': 10.5, 'pl_trandurh': 3.2, ...},
    {'pl_orbper': 5.2, 'pl_trandurh': 2.1, ...},
]
predictions, probabilities = predictor.predict_tess(observations)
```

## 🎯 실행 결과

```
✅ TESS 모델 로드: tess_model_20251005_194536.pkl
✅ Kepler+K2 모델 로드: kepler_k2_model_20251005_194536.pkl

📊 예측 결과:
  분류: ✅ 외계행성 후보
  후보 확률: 94.76%
  비후보 확률: 5.24%
```

## 📁 생성된 파일

- ✅ `sample_input.csv` - 예제 입력 파일 (수정해서 사용)
- ✅ `predictions_output.csv` - 예측 결과 저장

## 📚 자세한 문서

- 📖 **상세 가이드**: `PREDICTION_GUIDE.md`
- 💡 **사용 예제**: `predict_exoplanet.py` 파일 참고

## ⚡ 빠른 테스트

```bash
# 전체 예제 실행 (샘플 파일 생성 포함)
conda run -n nasa python predict_exoplanet.py
```

## 🔍 주요 변수 설명

| 변수         | 의미      | 일반적 범위      |
| ------------ | --------- | ---------------- |
| `pl_orbper`  | 궤도 주기 | 1-365일          |
| `pl_rade`    | 행성 크기 | 0.5-15 (지구=1)  |
| `pl_trandep` | 통과 깊이 | 100-10000 ppm    |
| `st_teff`    | 별 온도   | 3000-7000 K      |
| `st_rad`     | 별 크기   | 0.5-2.0 (태양=1) |

## ✨ 모델 성능

- **정확도**: 87.16%
- **F1 Score**: 91.02%
- **후보 탐지율**: 94%

---

**Happy Planet Hunting! 🌟🪐**
