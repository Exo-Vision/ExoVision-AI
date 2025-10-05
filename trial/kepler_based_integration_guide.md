# Kepler 기준 데이터셋 통합 가이드

**분석 날짜**: 2025년 10월 5일  
**데이터셋**: Kepler (9,564행 × 119컬럼), K2 (4,004행 × 201컬럼), TESS (7,703행 × 57컬럼)

---

## 📊 매핑 통계 요약

| 항목 | 개수 | 비율 |
|------|------|------|
| **총 Kepler 기준 컬럼** | 24개 | 100% |
| **K2 매핑 가능** | 23개 | 95.8% |
| **TESS 매핑 가능** | 14개 | 58.3% |
| **세 데이터셋 모두 존재** | 14개 | 58.3% |

---

## 🔄 단위 변환이 필요한 컬럼 (5개)

### 1. `koi_disposition` (정답 레이블) ⭐ **매우 중요**

| 데이터셋 | 컬럼명 | 값 형식 |
|---------|-------|--------|
| **Kepler** | `koi_disposition` | `CONFIRMED` / `FALSE POSITIVE` / `CANDIDATE` |
| **K2** | `pl_k2_disposition` | `CONFIRMED` / `FALSE POSITIVE` / `CANDIDATE` |
| **TESS** | `tfopwg_disp` | `CP` / `FP` / `KP` / `PC` |

**변환 규칙 (TESS → Kepler)**:
```python
tess_to_kepler = {
    'CP': 'CONFIRMED',      # Confirmed Planet
    'KP': 'CONFIRMED',      # Known Planet
    'FP': 'FALSE POSITIVE', # False Positive
    'PC': 'CANDIDATE'       # Planet Candidate
}
```

---

### 2. `koi_time0bk` (통과 중심 시각)

| 데이터셋 | 컬럼명 | 단위 |
|---------|-------|-----|
| **Kepler** | `koi_time0bk` | **BJD - 2454833.0** |
| **K2** | `pl_tranmid` | BJD |
| **TESS** | `pl_tranmid` | BJD |

**변환 규칙**:
```python
# Kepler → 표준 BJD
kepler_standard_bjd = df['koi_time0bk'] + 2454833.0
```

---

### 3. `koi_duration` (통과 지속시간)

| 데이터셋 | 컬럼명 | 단위 |
|---------|-------|-----|
| **Kepler** | `koi_duration` | **hours** |
| **K2** | `pl_trandur` | **days** ⚠️ |
| **TESS** | `pl_trandurh` | **hours** |

**변환 규칙**:
```python
# K2 → hours
k2_duration_hours = df['pl_trandur'] * 24
```

---

### 4. `koi_depth` (통과 깊이) ⚠️ **매우 중요**

| 데이터셋 | 컬럼명 | 단위 |
|---------|-------|-----|
| **Kepler** | `koi_depth` | **ppm** |
| **K2** | `pl_trandep` | **% (percent)** ⚠️ |
| **TESS** | `pl_trandep` | **ppm** |

**변환 규칙**:
```python
# K2 → ppm
k2_depth_ppm = df['pl_trandep'] * 10000
```

---

### 5. `koi_kepmag` (Kepler 등급)

| 데이터셋 | 컬럼명 | 단위 |
|---------|-------|-----|
| **Kepler** | `koi_kepmag` | magnitude |
| **K2** | `sy_kepmag` | magnitude |
| **TESS** | ❌ 없음 | N/A (TESS는 `st_tmag` 사용) |

**처리 방법**:
- TESS는 `koi_kepmag` 컬럼을 `NaN`으로 채우기
- 또는 별도로 `tess_mag` 컬럼 추가

---

## 📋 Kepler 기준 컬럼 매핑 전체 테이블

### 1️⃣ 기본 정보

| Kepler 컬럼 | K2 컬럼 | TESS 컬럼 | 의미 | 단위 | 변환 |
|------------|---------|----------|------|------|------|
| `kepoi_name` | `epic_candname` | `toi` | 행성 후보 이름 | - | ✅ 없음 |

---

### 2️⃣ 정답 레이블 (타겟)

| Kepler 컬럼 | K2 컬럼 | TESS 컬럼 | 의미 | 단위 | 변환 |
|------------|---------|----------|------|------|------|
| `koi_disposition` | `pl_k2_disposition` | `tfopwg_disp` | 행성 판별 결과 | categorical | 🔄 TESS 변환 필요 |

---

### 3️⃣ 위치 정보

| Kepler 컬럼 | K2 컬럼 | TESS 컬럼 | 의미 | 단위 | 변환 |
|------------|---------|----------|------|------|------|
| `ra` | `ra` | `ra` | 적경 | degrees | ✅ 없음 |
| `dec` | `dec` | `dec` | 적위 | degrees | ✅ 없음 |

---

### 4️⃣ 궤도 파라미터

| Kepler 컬럼 | K2 컬럼 | TESS 컬럼 | 의미 | 단위 | 변환 |
|------------|---------|----------|------|------|------|
| `koi_period` | `pl_orbper` | `pl_orbper` | 궤도 주기 | days | ✅ 없음 |
| `koi_time0bk` | `pl_tranmid` | `pl_tranmid` | 통과 중심 시각 | BJD | 🔄 Kepler 변환 필요 |
| `koi_eccen` | `pl_orbeccen` | ❌ 없음 | 궤도 이심률 | 0~1 | ⚠️ TESS 누락 |
| `koi_longp` | `pl_orblper` | ❌ 없음 | 근점 인수 | degrees | ⚠️ TESS 누락 |
| `koi_incl` | `pl_orbincl` | ❌ 없음 | 궤도 경사각 | degrees | ⚠️ TESS 누락 |
| `koi_impact` | `pl_imppar` | ❌ 없음 | 충격 계수 | R_star | ⚠️ TESS 누락 |
| `koi_sma` | `pl_orbsmax` | ❌ 없음 | 궤도 반장축 | AU | ⚠️ TESS 누락 |

---

### 5️⃣ 통과 파라미터

| Kepler 컬럼 | K2 컬럼 | TESS 컬럼 | 의미 | 단위 | 변환 |
|------------|---------|----------|------|------|------|
| `koi_duration` | `pl_trandur` | `pl_trandurh` | 통과 지속시간 | hours | 🔄 K2 변환 필요 |
| `koi_depth` | `pl_trandep` | `pl_trandep` | 통과 깊이 | ppm | 🔄 K2 변환 필요 ⚠️ |
| `koi_ingress` | ❌ 없음 | ❌ 없음 | 진입 시간 | hours | ⚠️ K2, TESS 누락 |

---

### 6️⃣ 행성 물리량

| Kepler 컬럼 | K2 컬럼 | TESS 컬럼 | 의미 | 단위 | 변환 |
|------------|---------|----------|------|------|------|
| `koi_prad` | `pl_rade` | `pl_rade` | 행성 반지름 | R_Earth | ✅ 없음 |
| `koi_insol` | `pl_insol` | `pl_insol` | 행성 복사 플럭스 | F_Earth | ✅ 없음 |
| `koi_teq` | `pl_eqt` | `pl_eqt` | 행성 평형 온도 | Kelvin | ✅ 없음 |

---

### 7️⃣ 항성 물리량

| Kepler 컬럼 | K2 컬럼 | TESS 컬럼 | 의미 | 단위 | 변환 |
|------------|---------|----------|------|------|------|
| `koi_srad` | `st_rad` | `st_rad` | 항성 반지름 | R_Sun | ✅ 없음 |
| `koi_smass` | `st_mass` | ❌ 없음 | 항성 질량 | M_Sun | ⚠️ TESS 누락 |
| `koi_sage` | `st_age` | ❌ 없음 | 항성 나이 | Gyr | ⚠️ TESS 누락 |
| `koi_steff` | `st_teff` | `st_teff` | 항성 유효 온도 | Kelvin | ✅ 없음 |
| `koi_slogg` | `st_logg` | `st_logg` | 항성 표면 중력 | log10(cm/s²) | ✅ 없음 |
| `koi_smet` | `st_met` | ❌ 없음 | 항성 금속성 | [Fe/H] | ⚠️ TESS 누락 |

---

### 8️⃣ 광도 측정

| Kepler 컬럼 | K2 컬럼 | TESS 컬럼 | 의미 | 단위 | 변환 |
|------------|---------|----------|------|------|------|
| `koi_kepmag` | `sy_kepmag` | ❌ 없음 | Kepler 등급 | magnitude | ⚠️ TESS 누락 |

---

## ⚠️ 데이터셋별 누락 컬럼 정리

### K2 데이터셋에 없는 컬럼 (1개)
- `koi_ingress` (진입 시간) - Kepler에만 있음

### TESS 데이터셋에 없는 컬럼 (10개)
1. `koi_eccen` (궤도 이심률)
2. `koi_longp` (근점 인수)
3. `koi_incl` (궤도 경사각)
4. `koi_impact` (충격 계수)
5. `koi_sma` (궤도 반장축)
6. `koi_ingress` (진입 시간)
7. `koi_smass` (항성 질량)
8. `koi_sage` (항성 나이)
9. `koi_smet` (항성 금속성)
10. `koi_kepmag` (Kepler 등급)

**처리 방법**: 누락된 컬럼은 `NaN`으로 채우기

---

## 💡 통합 전략 및 권장사항

### 1. 정답 레이블 통일 ⭐
```python
# TESS 정답 변환
label_mapping = {
    'CP': 'CONFIRMED',
    'KP': 'CONFIRMED',
    'FP': 'FALSE POSITIVE',
    'PC': 'CANDIDATE'
}
df_tess['koi_disposition'] = df_tess['tfopwg_disp'].map(label_mapping)
```

### 2. 단위 변환
```python
# K2 통과 깊이: % → ppm
df_k2['koi_depth'] = df_k2['pl_trandep'] * 10000

# K2 통과 지속시간: days → hours
df_k2['koi_duration'] = df_k2['pl_trandur'] * 24

# Kepler BJD 표준화
df_kepler['koi_time0bk'] = df_kepler['koi_time0bk'] + 2454833.0
```

### 3. 누락 컬럼 처리
```python
# TESS에 없는 컬럼 추가
missing_cols = ['koi_eccen', 'koi_longp', 'koi_incl', 'koi_impact', 
                'koi_sma', 'koi_ingress', 'koi_smass', 'koi_sage', 
                'koi_smet', 'koi_kepmag']
for col in missing_cols:
    df_tess[col] = np.nan
```

### 4. 에러 컬럼 통일
```python
# 모든 에러 컬럼을 *_error 형식으로 통합
# 예: koi_period_err1, koi_period_err2 → koi_period_error
```

### 5. Limit Flag 보존
```python
# K2, TESS의 limit flag 유지
# 예: pl_orbper_limit_flag → koi_period_limit_flag
```

### 6. 데이터셋 출처 표시
```python
df_kepler['source'] = 'Kepler'
df_k2['source'] = 'K2'
df_tess['source'] = 'TESS'
```

---

## 🎯 최종 통합 데이터셋 구조

| 컬럼 카테고리 | Kepler | K2 | TESS | 통합 후 |
|-------------|--------|----|----|---------|
| 기본 정보 | ✅ | ✅ | ✅ | ✅ |
| 정답 레이블 | ✅ | ✅ | 🔄 변환 | ✅ |
| 위치 정보 | ✅ | ✅ | ✅ | ✅ |
| 궤도 파라미터 (7개) | ✅ | ✅ | ⚠️ 5개 누락 | ✅ (NaN 포함) |
| 통과 파라미터 (3개) | ✅ | 🔄 변환 | 🔄 변환 | ✅ |
| 행성 물리량 (3개) | ✅ | ✅ | ✅ | ✅ |
| 항성 물리량 (6개) | ✅ | ✅ | ⚠️ 3개 누락 | ✅ (NaN 포함) |
| 광도 측정 (1개) | ✅ | ✅ | ❌ | ✅ (NaN 포함) |

**예상 통합 데이터셋 크기**: 21,271 행 (9,564 + 4,004 + 7,703)

---

## 📝 주의사항

1. **K2 통과 깊이 단위**: 반드시 % × 10,000으로 ppm 변환
2. **TESS 정답 레이블**: CP/KP/FP/PC를 Kepler 형식으로 변환
3. **Kepler BJD**: 2454833.0을 더해서 표준 BJD로 변환
4. **누락 컬럼**: TESS는 10개 컬럼 누락 (NaN 처리)
5. **에러 컬럼**: 모든 데이터셋의 err1, err2를 통합하여 _error로 통일
6. **Limit Flag**: K2, TESS의 limit flag 보존 필요
7. **데이터셋 출처**: source 컬럼으로 원본 데이터셋 추적 가능하도록

---

생성일: 2025년 10월 5일
