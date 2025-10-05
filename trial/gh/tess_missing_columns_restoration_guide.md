# TESS 누락 컬럼 중요도 분석 및 복원 전략

**분석 날짜**: 2025년 10월 5일  
**목적**: TESS에 누락된 10개 컬럼의 외계행성 판별 중요도 분석 및 복원 방법 제시

---

## 📊 핵심 요약

### TESS에 누락된 10개 컬럼

| 중요도 | 컬럼 | K2 데이터 | 복원 방법 | 우선순위 |
|-------|------|----------|----------|---------|
| ⭐⭐⭐⭐⭐ | `koi_smass` (항성 질량) | ✅ 52.5% | 🔢 계산 | **최우선** |
| ⭐⭐⭐⭐⭐ | `koi_sma` (궤도 반장축) | ✅ 20.5% | 🔢 계산 | **최우선** |
| ⭐⭐⭐⭐⭐ | `koi_incl` (궤도 경사각) | ✅ 24.9% | 🔢 계산 | **최우선** |
| ⭐⭐⭐⭐ | `koi_impact` (충격 계수) | ✅ 37.2% | 🔢 계산 | 높음 |
| ⭐⭐⭐ | `koi_eccen` (궤도 이심률) | ✅ 10.7% | 📈 통계적 | 중간 |
| ⭐⭐⭐ | `koi_smet` (항성 금속성) | ✅ 42.4% | 📈 통계적 | 중간 |
| ⭐⭐ | `koi_ingress` (진입 시간) | ❌ | 🔢 계산 | 낮음 |
| ⭐⭐ | `koi_sage` (항성 나이) | ✅ 7.9% | 📈 기본값 | 낮음 |
| ⭐ | `koi_longp` (근점 인수) | ✅ 6.1% | 📈 기본값 | 매우 낮음 |
| ⭐ | `koi_kepmag` (Kepler 등급) | ✅ 99.6% | ❌ NaN | 매우 낮음 |

---

## 🎯 외계행성 판별에 중요한 컬럼 (최우선 복원)

### 1. `koi_smass` (항성 질량) ⭐⭐⭐⭐⭐

**중요도**: 매우 높음  
**K2 데이터**: ✅ 있음 (52.5%)  
**복원 방법**: 🔢 **계산 가능**

**왜 중요한가?**
- 궤도 반장축 계산의 필수 요소 (케플러 제3법칙)
- 행성 질량 추정에 사용
- 거주 가능 영역(HZ) 계산에 영향

**계산 방법**:
```python
# 항성 질량 = 항성 반지름² × 10^(logg_star - logg_sun)
# logg_sun = 4.44

M_star = (R_star ** 2) * 10 ** (logg_star - 4.44)
```

**필요한 TESS 컬럼**:
- ✅ `st_rad` (항성 반지름) - TESS에 있음
- ✅ `st_logg` (항성 표면 중력) - TESS에 있음

**정확도**: 높음 (항성 모델 기반, ±10~20% 오차)

---

### 2. `koi_sma` (궤도 반장축) ⭐⭐⭐⭐⭐

**중요도**: 매우 높음  
**K2 데이터**: ✅ 있음 (20.5%)  
**복원 방법**: 🔢 **계산 가능**

**왜 중요한가?**
- **거주 가능 영역 판별**의 핵심 파라미터
- 행성 복사 플럭스 계산 (이미 TESS에 `pl_insol` 있음)
- 궤도 안정성 분석

**계산 방법** (케플러 제3법칙):
```python
# a³ = G·M_star·P² / (4π²)
# 단순화: a [AU] = (P[years]² × M_star[M_sun])^(1/3)

P_years = orbital_period_days / 365.25
a_AU = (P_years ** 2 * M_star) ** (1/3)
```

**필요한 TESS 컬럼**:
- ✅ `pl_orbper` (궤도 주기) - TESS에 있음
- 🔢 `koi_smass` (항성 질량) - 위에서 계산

**정확도**: 높음 (케플러 법칙, ±5~10% 오차)

---

### 3. `koi_incl` (궤도 경사각) ⭐⭐⭐⭐⭐

**중요도**: 매우 높음  
**K2 데이터**: ✅ 있음 (24.9%)  
**복원 방법**: 🔢 **계산 가능 (근사)**

**왜 중요한가?**
- **행성 반지름 계산의 필수** (통과 깊이 → 반지름 변환)
- 통과 기하학 이해
- 실제 행성 질량 계산 (M·sin(i) → M)

**계산 방법**:
```python
# 통과하는 행성은 경사각이 ~90도에 가까움
# 정확한 계산: cos(i) ≈ b·R_star/a
# 단순 근사: i ≈ 89도 (대부분 통과 행성)

orbital_inclination = 89.0  # degrees
```

**정확도**: 중간 (실제 87~90도 범위, 평균값 사용)

**더 정확한 계산** (통과 지속시간 사용):
```python
# T_dur = (P/π) × arcsin(R_star/a × sqrt((1+k)² - b²) / sin(i))
# 여기서 k = R_planet/R_star, b = impact parameter
# 역계산으로 i 추정 가능 (복잡)
```

---

### 4. `koi_impact` (충격 계수) ⭐⭐⭐⭐

**중요도**: 높음  
**K2 데이터**: ✅ 있음 (37.2%)  
**복원 방법**: 🔢 **계산 가능**

**왜 중요한가?**
- 통과 중심성 (행성이 항성 중심을 지나는지)
- 통과 깊이 해석에 중요
- 통과 대칭성 판단

**계산 방법**:
```python
# b = (a/R_star) × cos(i)
# 1 AU = 215.032 R_sun

a_in_Rstar = (a_AU * 215.032) / R_star_Rsun
impact_parameter = a_in_Rstar * np.cos(np.radians(inclination))
```

**필요한 컬럼**:
- 🔢 `koi_sma` (궤도 반장축)
- ✅ `st_rad` (항성 반지름)
- 🔢 `koi_incl` (궤도 경사각)

**정확도**: 중간 (경사각 근사에 의존)

---

## 📈 통계적 추정 컬럼

### 5. `koi_eccen` (궤도 이심률) ⭐⭐⭐

**중요도**: 높음  
**K2 데이터**: ✅ 있음 (10.7% - 매우 적음)  
**복원 방법**: 📈 **통계적 추정**

**왜 중요한가?**
- 궤도 형태 (원형 vs 타원형)
- 거주 가능성 (이심률 높으면 온도 변화 큼)
- 행성 형성 역사

**권장 처리**:
```python
# 대부분 통과 행성은 낮은 이심률
# 보수적 접근: 원형 궤도 가정
eccentricity = 0.0

# 또는 통계적 분포 (Rayleigh 분포, σ=0.05)
eccentricity = np.random.rayleigh(0.05)
```

**K2 데이터 활용**:
- K2에서도 10.7%만 있음 (측정 어려움)
- 측정값 있는 경우 사용, 없으면 0.0

---

### 6. `koi_smet` (항성 금속성) ⭐⭐⭐

**중요도**: 높음  
**K2 데이터**: ✅ 있음 (42.4%)  
**복원 방법**: 📈 **통계적 추정**

**왜 중요한가?**
- 행성 형성 확률 (금속성 높을수록 행성 많음)
- 암석 행성 vs 가스 행성 구분
- 거짓 양성 판별

**권장 처리**:
```python
# 태양 금속성 가정
metallicity = 0.0  # [Fe/H] = 0 (태양)

# 또는 통계적 분포 (정규분포, μ=0, σ=0.2)
metallicity = np.random.normal(0.0, 0.2)
```

**K2 데이터 활용**:
- K2에서 42.4% 있음
- 가능하면 K2 데이터 사용

---

## 🔢 전체 복원 코드

```python
import pandas as pd
import numpy as np

def restore_tess_missing_columns(df_tess):
    """
    TESS 데이터셋의 누락 컬럼을 복원
    
    Parameters:
    -----------
    df_tess : DataFrame
        TESS merged 데이터셋
        
    Returns:
    --------
    DataFrame
        누락 컬럼이 복원된 데이터셋
    """
    
    df = df_tess.copy()
    
    print("=" * 80)
    print("TESS 누락 컬럼 복원 시작")
    print("=" * 80)
    
    # ========== 1. 항성 질량 계산 (최우선) ==========
    if 'st_rad' in df.columns and 'st_logg' in df.columns:
        logg_sun = 4.44
        df['koi_smass'] = (df['st_rad'] ** 2) * 10 ** (df['st_logg'] - logg_sun)
        
        # 이상치 제거 (0.1~10 태양질량 범위)
        df.loc[(df['koi_smass'] < 0.1) | (df['koi_smass'] > 10), 'koi_smass'] = np.nan
        
        valid = df['koi_smass'].notna().sum()
        print(f"✅ 항성 질량 계산 완료: {valid}/{len(df)} ({valid/len(df)*100:.1f}%)")
    else:
        print("❌ 항성 질량 계산 실패: 필요한 컬럼 없음")
    
    # ========== 2. 궤도 반장축 계산 (최우선) ==========
    if 'pl_orbper' in df.columns and 'koi_smass' in df.columns:
        P_years = df['pl_orbper'] / 365.25
        df['koi_sma'] = (P_years ** 2 * df['koi_smass']) ** (1/3)
        
        # 이상치 제거 (0.001~100 AU 범위)
        df.loc[(df['koi_sma'] < 0.001) | (df['koi_sma'] > 100), 'koi_sma'] = np.nan
        
        valid = df['koi_sma'].notna().sum()
        print(f"✅ 궤도 반장축 계산 완료: {valid}/{len(df)} ({valid/len(df)*100:.1f}%)")
    else:
        print("❌ 궤도 반장축 계산 실패: 필요한 컬럼 없음")
    
    # ========== 3. 궤도 경사각 (근사) ==========
    # 통과하는 행성은 대부분 i ≈ 89도
    df['koi_incl'] = 89.0  # degrees
    print(f"⚠️  궤도 경사각 기본값 설정: 89도 (통과 행성 근사)")
    
    # ========== 4. 충격 계수 계산 ==========
    if 'koi_sma' in df.columns and 'st_rad' in df.columns and 'koi_incl' in df.columns:
        # 1 AU = 215.032 R_sun
        AU_to_Rsun = 215.032
        a_in_Rstar = (df['koi_sma'] * AU_to_Rsun) / df['st_rad']
        df['koi_impact'] = a_in_Rstar * np.cos(np.radians(df['koi_incl']))
        
        # 충격 계수는 0~1 범위 (통과 행성)
        df.loc[(df['koi_impact'] < 0) | (df['koi_impact'] > 1), 'koi_impact'] = np.nan
        
        valid = df['koi_impact'].notna().sum()
        print(f"✅ 충격 계수 계산 완료: {valid}/{len(df)} ({valid/len(df)*100:.1f}%)")
    else:
        print("❌ 충격 계수 계산 실패: 필요한 컬럼 없음")
    
    # ========== 5. 궤도 이심률 (통계적) ==========
    df['koi_eccen'] = 0.0  # 원형 궤도 가정
    print(f"⚠️  궤도 이심률 기본값: 0.0 (원형 궤도 가정)")
    
    # ========== 6. 근점 인수 (기본값) ==========
    df['koi_longp'] = 90.0  # degrees
    print(f"⚠️  근점 인수 기본값: 90도")
    
    # ========== 7. 항성 금속성 (통계적) ==========
    df['koi_smet'] = 0.0  # 태양 금속성
    print(f"⚠️  항성 금속성 기본값: 0.0 (태양 금속성)")
    
    # ========== 8. 항성 나이 (기본값) ==========
    df['koi_sage'] = 5.0  # Gyr
    print(f"⚠️  항성 나이 기본값: 5 Gyr (태양 나이)")
    
    # ========== 9. 진입 시간 (추정) ==========
    if 'pl_trandurh' in df.columns:
        df['koi_ingress'] = df['pl_trandurh'] * 0.1  # 통과 시간의 10%
        valid = df['koi_ingress'].notna().sum()
        print(f"✅ 진입 시간 추정 완료: {valid}/{len(df)} ({valid/len(df)*100:.1f}%)")
    else:
        print("❌ 진입 시간 추정 실패: pl_trandurh 없음")
    
    # ========== 10. Kepler 등급 (NaN) ==========
    df['koi_kepmag'] = np.nan
    print(f"⚠️  Kepler 등급: NaN (TESS는 st_tmag 사용)")
    
    print("=" * 80)
    print("✅ TESS 누락 컬럼 복원 완료")
    print("=" * 80)
    
    return df
```

---

## 📊 K2 데이터 활용 전략

### K2 컬럼 데이터 가용성

| 컬럼 | K2 컬럼명 | 데이터 비율 | 활용 전략 |
|------|----------|----------|---------|
| `koi_eccen` | `pl_orbeccen` | 10.7% | 있으면 사용, 없으면 0.0 |
| `koi_longp` | `pl_orblper` | 6.1% | 있으면 사용, 없으면 90° |
| `koi_incl` | `pl_orbincl` | 24.9% | 있으면 사용, 없으면 89° |
| `koi_impact` | `pl_imppar` | 37.2% | 있으면 사용, 없으면 계산 |
| `koi_sma` | `pl_orbsmax` | 20.5% | 있으면 사용, 없으면 계산 |
| `koi_smass` | `st_mass` | 52.5% | 있으면 사용, 없으면 계산 |
| `koi_sage` | `st_age` | 7.9% | 있으면 사용, 없으면 5 Gyr |
| `koi_smet` | `st_met` | 42.4% | 있으면 사용, 없으면 0.0 |
| `koi_kepmag` | `sy_kepmag` | 99.6% | K2에서 사용 |

**권장 전략**:
1. Kepler 데이터: 실측값 그대로 사용
2. K2 데이터: 있으면 실측값, 없으면 계산/추정
3. TESS 데이터: 모두 계산/추정

---

## ⚠️ 데이터 품질 주의사항

### 1. 계산값 vs 측정값 구분

```python
# 데이터 출처 표시
df['koi_smass_source'] = 'calculated'  # TESS
df['koi_smass_source'] = 'measured'    # Kepler/K2 (있는 경우)

# 또는 별도 컬럼
df['is_calculated'] = True  # TESS
df['is_calculated'] = False  # Kepler/K2
```

### 2. 정확도 경고

| 컬럼 | 계산 방법 | 예상 오차 | 주의사항 |
|------|---------|---------|---------|
| `koi_smass` | 항성 모델 | ±10~20% | 젊은/늙은 항성에서 오차 증가 |
| `koi_sma` | 케플러 법칙 | ±5~10% | 항성 질량 오차에 의존 |
| `koi_incl` | 근사 (89°) | ±2~3° | 실제는 87~90° 범위 |
| `koi_impact` | 기하학 | ±0.1~0.2 | 경사각 근사에 의존 |
| `koi_eccen` | 가정 (0.0) | ±0.1 | 일부 행성은 높은 이심률 |

### 3. 머신러닝 사용 시 권장사항

```python
# Feature 중요도 분석 시 계산 컬럼 별도 표시
calculated_features = ['koi_smass', 'koi_sma', 'koi_incl', 'koi_impact']

# 또는 계산 컬럼 제외하고 학습
real_features = [col for col in features if col not in calculated_features]

# 또는 계산 컬럼에 낮은 가중치 부여
feature_weights = {
    'koi_smass': 0.5,  # 계산값
    'koi_period': 1.0,  # 실측값
}
```

---

## 📈 최종 권장 통합 전략

### 1단계: Kepler 데이터 (9,564 행)
- 모든 컬럼 그대로 사용 (실측값)

### 2단계: K2 데이터 (4,004 행)
- 있는 컬럼: 그대로 사용
- 없는 컬럼: 계산 또는 추정

### 3단계: TESS 데이터 (7,703 행)
- **최우선 복원** (계산):
  - `koi_smass` ← `st_rad`, `st_logg`
  - `koi_sma` ← `pl_orbper`, `koi_smass`
  - `koi_incl` ← 89° (근사)
  - `koi_impact` ← `koi_sma`, `st_rad`, `koi_incl`
  
- **통계적 추정**:
  - `koi_eccen` ← 0.0
  - `koi_smet` ← 0.0
  - `koi_sage` ← 5.0 Gyr
  - `koi_longp` ← 90°
  
- **제외**:
  - `koi_kepmag` ← NaN

### 4단계: 통합 데이터셋 생성
- 총 21,271 행
- 모든 Kepler 기준 컬럼 포함
- `source` 컬럼으로 원본 추적
- `is_calculated` 컬럼으로 계산값 표시

---

생성일: 2025년 10월 5일
