# 외계행성 데이터셋 에러 컬럼 통합 가이드

## 📚 에러 컬럼의 의미

외계행성 데이터셋의 각 측정값에는 **불확실성(uncertainty)**을 나타내는 에러 컬럼이 함께 제공됩니다.

### 🔹 err1 (Upper Uncertainty) - 상위 불확실성
- 측정값의 **상위 불확실성** (양의 오차)
- 실제 값이 측정값보다 **클 가능성**
- **예시**: `pl_orbper = 10.5`, `pl_orbpererr1 = 0.2` 
  - → 실제 값은 **10.5 ~ 10.7** 사이

### 🔹 err2 (Lower Uncertainty) - 하위 불확실성
- 측정값의 **하위 불확실성** (음의 오차)
- 실제 값이 측정값보다 **작을 가능성**
- **예시**: `pl_orbper = 10.5`, `pl_orbpererr2 = -0.3`
  - → 실제 값은 **10.2 ~ 10.5** 사이

### 🔹 lim (Limit Flag) - 제한 플래그
- **0**: 정상 측정값
- **1**: 상한값 (Upper Limit) - 실제 값이 이보다 **작음**
- **-1**: 하한값 (Lower Limit) - 실제 값이 이보다 **큼**
- **예시**: `pl_masse = 100`, `pl_masselim = 1` 
  - → 행성 질량은 **100 이하**

---

## 📊 데이터셋별 에러 컬럼 구조

### Kepler 데이터셋
| 특징 | 내용 |
|------|------|
| 에러 접미사 | `_err1`, `_err2` |
| 제한 플래그 | ❌ 없음 |
| 예시 | `koi_period`, `koi_period_err1`, `koi_period_err2` |
| 특징 | 비대칭 에러만 제공 |

### K2 데이터셋
| 특징 | 내용 |
|------|------|
| 에러 접미사 | `err1`, `err2` |
| 제한 플래그 | ✅ 있음 (`lim`) |
| 예시 | `pl_orbper`, `pl_orbpererr1`, `pl_orbpererr2`, `pl_orbperlim` |
| 특징 | 비대칭 에러 + 제한 플래그 |

### TESS 데이터셋
| 특징 | 내용 |
|------|------|
| 에러 접미사 | `err1`, `err2` |
| 제한 플래그 | ✅ 있음 (`lim`) |
| 예시 | `pl_orbper`, `pl_orbpererr1`, `pl_orbpererr2`, `pl_orbperlim` |
| 특징 | 비대칭 에러 + 제한 플래그 |

---

## 🎯 5가지 통합 전략

### 전략 1: 단순 평균 에러 ⭐ (가장 많이 사용)

**설명**: 상위/하위 에러의 절대값 평균

**공식**:
```python
error = (|err1| + |err2|) / 2
```

**적용 대상**: 대칭에 가까운 에러, 일반적인 물리량

**장점**: 
- ✅ 단순하고 직관적
- ✅ 계산 빠름
- ✅ 해석 용이

**단점**: 
- ❌ 비대칭성 정보 손실

**예시**: 궤도 주기, 온도, 반지름

---

### 전략 2: 가중 평균 (신뢰도 기반)

**설명**: 에러가 작을수록 높은 가중치 부여

**공식**:
```python
value = Σ(vi / σi²) / Σ(1 / σi²)
```

**적용 대상**: 여러 논문의 측정값을 통합할 때

**장점**: 
- ✅ 정확한 측정에 더 큰 영향

**단점**: 
- ❌ 복잡한 계산

---

### 전략 3: 최대 에러 사용 (보수적 접근)

**설명**: 더 큰 에러를 사용하는 보수적 접근

**공식**:
```python
error = max(|err1|, |err2|)
```

**적용 대상**: 안전성이 중요한 경우

**장점**: 
- ✅ 보수적, 안전한 추정

**단점**: 
- ❌ 불확실성 과대평가

**예시**: 행성 거주 가능성 판단, 항성 온도/질량

---

### 전략 4: 제한 플래그 고려

**설명**: `lim` 값에 따라 다른 처리

**로직**:
```python
if lim == 1:
    # 상한값 - 실제 값은 이보다 작음
    handle_as_upper_limit()
elif lim == -1:
    # 하한값 - 실제 값은 이보다 큼
    handle_as_lower_limit()
else:  # lim == 0
    # 정상 측정값
    use_normal_value()
```

**적용 대상**: K2, TESS의 `lim` 컬럼이 있는 경우

**장점**: 
- ✅ 물리적 의미 보존

**단점**: 
- ❌ 복잡한 로직

**예시**: 행성 질량 (측정 불가능한 경우 많음)

---

### 전략 5: 분리 보존 (정밀 분석용)

**설명**: 비대칭 에러를 그대로 유지

**결과**:
```
value, error_upper, error_lower (3개 컬럼)
```

**적용 대상**: 정밀한 분석이 필요한 경우

**장점**: 
- ✅ 정보 손실 없음
- ✅ 비대칭성 보존

**단점**: 
- ❌ 컬럼 수 증가

**예시**: 논문 작성, 궤도 이심률, 항성 질량

---

## 💡 컬럼별 권장 통합 방법

| 컬럼 | 특성 | 권장 전략 | 이유 |
|------|------|-----------|------|
| **궤도 주기** (`pl_orbper`) | 매우 정확한 측정 | ⭐ 전략 1 (평균) | 측정 정확도 높고 대칭적 |
| **행성 반지름** (`pl_rade`) | 중간 정확도 | ⭐ 전략 1 (평균) | 일반적으로 대칭 |
| **행성 질량** (`pl_masse`) | 측정 어려움 | 전략 4 (lim 고려) | 상한값만 알려진 경우 많음 |
| **궤도 이심률** (`pl_orbeccen`) | 0~1 범위, 비대칭 | 전략 5 (분리) or 전략 3 (최대) | 경계값 근처 비대칭 |
| **평형 온도** (`pl_eqt`) | 계산값 | ⭐ 전략 1 (평균) | 일반적으로 대칭 |
| **항성 온도** (`st_teff`) | 스펙트럼 분석 | 전략 3 (최대) | 보수적 접근 권장 |
| **항성 질량** (`st_mass`) | 간접 추정 | 전략 3 (최대) or 전략 5 (분리) | 큰 불확실성 |
| **항성 반지름** (`st_rad`) | 중간 정확도 | ⭐ 전략 1 (평균) | 일반적으로 대칭 |

---

## 💻 Python 구현 예제

### 기본 통합 함수

```python
import pandas as pd
import numpy as np

def merge_error_columns(df, base_col, strategy='average', dataset='k2'):
    """
    에러 컬럼을 하나로 통합하는 함수
    
    Parameters:
    -----------
    df : DataFrame
        데이터프레임
    base_col : str
        기본 컬럼명 (예: 'pl_orbper')
    strategy : str
        'average': 평균 에러
        'max': 최대 에러
        'separate': 분리 보존
    dataset : str
        'kepler' 또는 'k2'/'tess'
    """
    
    # 데이터셋별 에러 컬럼명 패턴
    if dataset == 'kepler':
        err1_col = base_col + '_err1'
        err2_col = base_col + '_err2'
        lim_col = None
    else:  # k2, tess
        err1_col = base_col + 'err1'
        err2_col = base_col + 'err2'
        lim_col = base_col + 'lim'
    
    # 컬럼 존재 여부 확인
    if err1_col not in df.columns or err2_col not in df.columns:
        return df
    
    # 전략별 처리
    if strategy == 'average':
        df[base_col + '_error'] = (
            df[err1_col].abs() + df[err2_col].abs()
        ) / 2
        
    elif strategy == 'max':
        df[base_col + '_error'] = np.maximum(
            df[err1_col].abs(),
            df[err2_col].abs()
        )
        
    elif strategy == 'separate':
        df[base_col + '_error_upper'] = df[err1_col].abs()
        df[base_col + '_error_lower'] = df[err2_col].abs()
    
    # Limit 플래그 처리 (K2/TESS만)
    if lim_col and lim_col in df.columns:
        df[base_col + '_limit_flag'] = df[lim_col]
    
    return df
```

### 사용 예제

```python
# K2 데이터 로드
df_k2 = pd.read_csv('datasets/k2.csv')

# 궤도 주기: 평균 에러 (가장 일반적)
df_k2 = merge_error_columns(df_k2, 'pl_orbper', strategy='average', dataset='k2')

# 행성 반지름: 평균 에러
df_k2 = merge_error_columns(df_k2, 'pl_rade', strategy='average', dataset='k2')

# 행성 질량: 최대 에러 (보수적)
df_k2 = merge_error_columns(df_k2, 'pl_masse', strategy='max', dataset='k2')

# 궤도 이심률: 분리 보존 (정밀 분석)
df_k2 = merge_error_columns(df_k2, 'pl_orbeccen', strategy='separate', dataset='k2')

# Kepler 데이터
df_kepler = pd.read_csv('datasets/kepler.csv')
df_kepler = merge_error_columns(df_kepler, 'koi_period', strategy='average', dataset='kepler')
```

---

## 📈 실제 데이터 예시

### K2 데이터 - 궤도 주기

```
행 1:
  측정값: 41.68864400 days
  상위 불확실성: +0.00335300 days
  하위 불확실성: -0.00341900 days
  실제 범위: 41.68522500 ~ 41.69199700 days
  
  → 평균 에러: ±0.00338600 days  ⭐ 권장
  → 최대 에러: ±0.00341900 days
```

### Kepler 데이터 - 궤도 주기

```
행 1:
  측정값: 9.48803557 days
  상위 불확실성: +0.0000277500 days
  하위 불확실성: -0.0000277500 days
  실제 범위: 9.48800782 ~ 9.48806332 days
  
  → 평균 에러: ±0.0000277500 days  ⭐ 권장
  → 최대 에러: ±0.0000277500 days
```

---

## ⚠️ 주의사항

1. **NaN 값 처리**: 에러 컬럼에 NaN이 있을 수 있음 → `fillna()` 또는 조건부 처리
2. **제한 플래그 해석**: `lim != 0`인 경우 실제 측정값이 아님
3. **비대칭 에러**: err1과 err2의 부호와 크기가 다를 수 있음
4. **단위 일관성**: 에러도 본 값과 같은 단위
5. **정보 손실**: 에러를 합치면 원본 정보 일부 손실 → 백업 권장
6. **데이터셋별 차이**: Kepler는 `lim` 없음, K2/TESS는 있음
7. **과학적 엄밀성**: 논문용 분석은 전략 5 (분리 보존) 권장

---

## ✅ 목적별 최종 권장사항

### 🎯 머신러닝/분류 모델용
- **권장 전략**: 전략 1 (평균) + 전략 4 (lim 고려)
- **이유**: 단순하고 해석 가능
- **추가**: 
  - `lim != 0`인 데이터는 제외 또는 별도 처리
  - 에러 컬럼을 feature로 추가 고려

### 🎯 탐색적 데이터 분석(EDA)용
- **권장 전략**: 전략 1 (평균)
- **이유**: 빠르고 직관적, 시각화 적합
- **추가**: 에러바 표시 시 평균 에러 사용

### 🎯 과학 논문/상세 분석용
- **권장 전략**: 전략 5 (분리 보존)
- **이유**: 정보 손실 없음, 비대칭 에러 중요
- **추가**: 에러 전파(error propagation) 계산 필요

### 🎯 프로덕션 시스템용
- **권장 전략**: 전략 3 (최대 에러)
- **이유**: 보수적 접근, 안전성 우선
- **추가**: 불확실성 높은 예측 회피

---

## 📝 데이터 전처리 체크리스트

- [ ] 에러 컬럼 존재 여부 확인
- [ ] NaN 값 처리 방법 결정
- [ ] 제한 플래그(`lim`) 처리 방법 결정
- [ ] 통합 전략 선택
- [ ] 원본 데이터 백업
- [ ] 통합 후 데이터 검증
- [ ] 컬럼명 정리 (원본 에러 컬럼 제거 여부)

---

생성일: 2025년 10월 5일
