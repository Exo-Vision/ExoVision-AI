# 모델 로드 및 예측 가이드

## 📋 개요
- **입력 형식**: 19개 원본 피처 (기본 천문 데이터)
- **출력 형식**: 3-클래스 분류 (CONFIRMED / FALSE POSITIVE / CANDIDATE)
- **모델 구조**: 2-모델 파이프라인 (확신도 기반 계층적 분류)

## 🚀 빠른 시작

### 1. 모델 로드 및 테스트
```bash
cd data_preprocessing/model_code
python load_and_predict_compatible.py
```

### 2. 출력 예시
```
✅ 전체 정확도: 0.7331 (73.31%)

예측 분포:
  CANDIDATE: 8,500개 (40.0%)
  CONFIRMED: 6,300개 (29.6%)
  FALSE POSITIVE: 6,400개 (30.4%)
```

## 📊 입력 데이터 형식

### 필수 19개 피처
```python
[
    'ra', 'dec',                    # 좌표
    'koi_period',                   # 궤도 주기
    'koi_time0bk',                  # 통과 시각
    'koi_impact',                   # 충격 파라미터
    'koi_duration',                 # 통과 지속시간
    'koi_depth',                    # 통과 깊이
    'koi_prad',                     # 행성 반지름
    'koi_teq',                      # 평형 온도
    'koi_insol',                    # 입사 플럭스
    'koi_sma',                      # 궤도 반장축
    'koi_eccen',                    # 이심률
    'koi_srad',                     # 항성 반지름
    'koi_smass',                    # 항성 질량
    'koi_sage',                     # 항성 나이
    'koi_steff',                    # 항성 유효온도
    'koi_slogg',                    # 항성 표면중력
    'koi_smet'                      # 항성 금속성
]
```

### CSV 파일 예시
```csv
ra,dec,koi_period,koi_time0bk,koi_impact,koi_duration,koi_depth,...
291.93423,48.141651,2.470613,131.512,0.146,2.95,0.00248,...
297.00733,48.134129,13.781239,133.349,0.969,2.68,0.00317,...
```

## 🔧 사용 방법

### 방법 1: 전체 데이터셋 예측
```python
import pandas as pd
from load_and_predict_compatible import predict_new_samples

# CSV 파일에서 예측
result = predict_new_samples('your_data.csv')
print(result.head())

# 결과 저장
result.to_csv('predictions.csv', index=False)
```

### 방법 2: 개별 샘플 예측
```python
import pandas as pd
from load_and_predict_compatible import predict, engineer_features
import numpy as np

# 단일 샘플 (딕셔너리)
sample = {
    'ra': 291.93423,
    'dec': 48.141651,
    'koi_period': 2.470613,
    'koi_time0bk': 131.512,
    # ... 나머지 15개 피처
}

# DataFrame으로 변환
X = pd.DataFrame([sample])

# 예측
predictions, confidences = predict(X)
print(f"예측: {predictions[0]}")
print(f"확신도: {confidences[0]:.4f}")
```

### 방법 3: 대용량 배치 예측
```python
import pandas as pd
from load_and_predict_compatible import predict
import numpy as np

# 대용량 CSV 청크 단위로 처리
chunk_size = 10000
results = []

for chunk in pd.read_csv('large_data.csv', chunksize=chunk_size):
    # 타겟 제거
    X = chunk.drop(['koi_disposition', 'kepoi_name'], axis=1, errors='ignore')
    
    # 예측
    preds, confs = predict(X)
    
    # 결과 저장
    chunk_result = pd.DataFrame({
        'kepoi_name': chunk['kepoi_name'] if 'kepoi_name' in chunk else range(len(preds)),
        'prediction': preds,
        'confidence': confs
    })
    results.append(chunk_result)

# 결과 통합
final_result = pd.concat(results, ignore_index=True)
final_result.to_csv('batch_predictions.csv', index=False)
```

## 📈 출력 형식

### 예측 결과 구조
```python
{
    'kepoi_name': 'K00001.01',          # 행성 ID
    'prediction': 'CONFIRMED',           # 예측 클래스
    'confidence': 0.9847                 # 확신도 (0~1)
}
```

### 클래스 설명
- **CONFIRMED**: 확인된 외계행성 (고확신도)
- **FALSE POSITIVE**: 거짓 양성 (비행성)
- **CANDIDATE**: 후보 (추가 관측 필요)

## 🎯 모델 성능

### 전체 정확도
- **목표**: 95.00%
- **현재**: 73.31%
- **모델 1** (CONFIRMED vs FALSE POSITIVE): 90.28%
- **모델 2** (CANDIDATE 판별): 75.28%

### 클래스별 성능
| 클래스 | Precision | Recall | F1-Score |
|--------|-----------|--------|----------|
| CANDIDATE | 0.67 | 0.76 | 0.71 |
| CONFIRMED | 0.74 | 0.70 | 0.72 |
| FALSE POSITIVE | 0.82 | 0.74 | 0.78 |

## ⚙️ 고급 설정

### 확신도 임계값 조정
```python
# config 파일 수정
import joblib

config = joblib.load('../../saved_models/config_YYYYMMDD_HHMMSS.pkl')
config['best_threshold'] = 0.92  # 기본값: 0.98

# 저장
joblib.dump(config, '../../saved_models/config_YYYYMMDD_HHMMSS.pkl')
```

### 피처 중요도 확인
```python
import joblib

model1 = joblib.load('../../saved_models/model1_ultra_YYYYMMDD_HHMMSS.pkl')

# CatBoost 피처 중요도
if hasattr(model1, 'estimators_'):
    for name, estimator in model1.estimators_:
        if hasattr(estimator, 'feature_importances_'):
            importances = estimator.feature_importances_
            print(f"{name} Top 5 features:")
            # ... (피처 이름 매핑 필요)
```

## 🐛 문제 해결

### 1. 모델 파일을 찾을 수 없음
```
❌ 저장된 모델이 없습니다!
```
**해결**: `model_code.py`를 먼저 실행하여 모델 학습

### 2. 피처 개수 불일치
```
ValueError: X has 18 features, but model expects 44
```
**해결**: 입력 데이터에 19개 필수 피처가 모두 있는지 확인

### 3. 메모리 부족
**해결**: 배치 예측 방법 3 사용 (청크 단위 처리)

## 📞 지원

- **이슈**: GitHub Issues
- **문서**: README.md
- **예제**: `example_prediction.py`

## 🔄 업데이트 로그

### v1.0 (2025-10-05)
- 초기 2-모델 시스템 구현
- SMOTE + 강화 앙상블
- 73.31% 정확도 달성
