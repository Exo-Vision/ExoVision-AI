# TESS Merged 데이터셋 컬럼 설명서

**데이터 크기**: 7,703 행 × 57 컬럼  
**원본 컬럼**: 87개 → **통합 후**: 57개 (30개 컬럼 축소)

---

## 📌 기본 정보

| 컬럼명 | 설명 | 단위 | 타입 |
|--------|------|------|------|
| `rowid` | 행 ID | - | 정수 |
| `toi` | TESS Object of Interest | - | 소수 |
| `toipfx` | TOI Prefix | - | 문자열 |
| `tid` | TESS Input Catalog ID | - | 정수 |
| `ctoi_alias` | TESS Input Catalog Alias | - | 문자열 |
| `pl_pnum` | Pipeline Signal ID | - | 정수 |

---

## 🎯 정답 (Target)

| 컬럼명 | 설명 | 단위 | 값 | 비고 |
|--------|------|------|-----|------|
| **`tfopwg_disp`** | **TFOPWG 판별 결과** (정답) | - | CP / FP / KP / PC | **머신러닝 타겟** |

### 정답 값 설명
- **CP** (Confirmed Planet): 확인된 행성 → `CONFIRMED`와 동일
- **FP** (False Positive): 거짓 양성 → `FALSE POSITIVE`와 동일  
- **KP** (Known Planet): 알려진 행성 → `CONFIRMED`로 통합 가능
- **PC** (Planet Candidate): 행성 후보 → `CANDIDATE`와 동일

---

## 📍 위치 정보

| 컬럼명 | 설명 | 단위 | 에러 컬럼 |
|--------|------|------|----------|
| `rastr` | 적경 (육십분법) | sexagesimal | - |
| **`ra`** | **적경** | degrees | `ra_error` |
| `decstr` | 적위 (육십분법) | sexagesimal | - |
| **`dec`** | **적위** | degrees | `dec_error` |

---

## 🌌 고유운동

| 컬럼명 | 설명 | 단위 | 에러 컬럼 | Limit Flag |
|--------|------|------|----------|------------|
| `st_pmra` | 고유운동 (RA) | mas/yr | `st_pmra_error` | `st_pmra_limit_flag` |
| `st_pmdec` | 고유운동 (Dec) | mas/yr | `st_pmdec_error` | `st_pmdec_limit_flag` |

---

## 🪐 행성 파라미터

### 궤도 파라미터

| 컬럼명 | 설명 | 단위 | 에러 컬럼 | 통합 전략 | Limit Flag |
|--------|------|------|----------|-----------|------------|
| `pl_tranmid` | 통과 중심 시각 | BJD | `pl_tranmid_error` | 평균 | `pl_tranmid_limit_flag` |
| **`pl_orbper`** | **궤도 주기** | days | `pl_orbper_error` | 평균 | `pl_orbper_limit_flag` |

### 통과 파라미터

| 컬럼명 | 설명 | 단위 | 에러 컬럼 | 통합 전략 | Limit Flag |
|--------|------|------|----------|-----------|------------|
| **`pl_trandurh`** | **통과 지속시간** | hours | `pl_trandurh_error` | 평균 | `pl_trandurh_limit_flag` |
| **`pl_trandep`** | **통과 깊이** | **ppm** | `pl_trandep_error` | 평균 | `pl_trandep_limit_flag` |

> ✅ TESS의 `pl_trandep`는 **ppm** 단위입니다 (Kepler와 동일)

### 행성 물리량

| 컬럼명 | 설명 | 단위 | 에러 컬럼 | 통합 전략 | Limit Flag |
|--------|------|------|----------|-----------|------------|
| **`pl_rade`** | **행성 반지름** | Earth radii | `pl_rade_error` | 평균 | `pl_rade_limit_flag` |
| **`pl_insol`** | **복사 플럭스** | Earth flux | `pl_insol_error` | 평균 | `pl_insol_limit_flag` |
| **`pl_eqt`** | **평형 온도** | Kelvin | `pl_eqt_error` | 평균 | `pl_eqt_limit_flag` |

---

## ⭐ 항성 파라미터

### 기본 물리량

| 컬럼명 | 설명 | 단위 | 에러 컬럼 | 통합 전략 | Limit Flag |
|--------|------|------|----------|-----------|------------|
| `st_tmag` | TESS 등급 | magnitude | `st_tmag_error` | 평균 | `st_tmag_limit_flag` |
| **`st_dist`** | **거리** | pc | `st_dist_error` | 평균 | `st_dist_limit_flag` |
| **`st_teff`** | **항성 유효 온도** | Kelvin | `st_teff_error` | 최대 (보수적) | `st_teff_limit_flag` |
| **`st_logg`** | **항성 표면 중력** | log10(cm/s²) | `st_logg_error` | 최대 (보수적) | `st_logg_limit_flag` |
| **`st_rad`** | **항성 반지름** | Solar radii | `st_rad_error` | 평균 | `st_rad_limit_flag` |

---

## 📅 메타데이터

| 컬럼명 | 설명 |
|--------|------|
| `toi_created` | TOI 생성 날짜 |
| `rowupdate` | 마지막 업데이트 날짜 |

---

## 📊 Symmetric Error Flag (대칭 에러 플래그)

> TESS 데이터셋에는 각 측정값마다 대칭 에러 여부를 나타내는 플래그가 있습니다.

| 플래그 컬럼 | 대응 측정값 | 의미 |
|------------|------------|------|
| `st_pmrasymerr` | `st_pmra` | 고유운동(RA) 에러가 대칭인지 |
| `st_pmdecsymerr` | `st_pmdec` | 고유운동(Dec) 에러가 대칭인지 |
| `pl_tranmidsymerr` | `pl_tranmid` | 통과 시각 에러가 대칭인지 |
| `pl_orbpersymerr` | `pl_orbper` | 궤도 주기 에러가 대칭인지 |
| `pl_trandurhsymerr` | `pl_trandurh` | 통과 지속시간 에러가 대칭인지 |
| `pl_trandepsymerr` | `pl_trandep` | 통과 깊이 에러가 대칭인지 |
| `pl_radesymerr` | `pl_rade` | 행성 반지름 에러가 대칭인지 |
| `pl_insolsymerr` | `pl_insol` | 복사 플럭스 에러가 대칭인지 |
| `pl_eqtsymerr` | `pl_eqt` | 평형 온도 에러가 대칭인지 |
| `st_tmagsymerr` | `st_tmag` | TESS 등급 에러가 대칭인지 |
| `st_distsymerr` | `st_dist` | 거리 에러가 대칭인지 |
| `st_teffsymerr` | `st_teff` | 항성 온도 에러가 대칭인지 |
| `st_loggsymerr` | `st_logg` | 항성 중력 에러가 대칭인지 |
| `st_radsymerr` | `st_rad` | 항성 반지름 에러가 대칭인지 |

---

## 💡 에러 컬럼 통합 규칙

### 평균 에러 (`_error`)
- **의미**: (상위 불확실성 + 하위 불확실성) / 2
- **사용**: 대부분의 측정값
- **예시**: `pl_orbper_error` = (|pl_orbpererr1| + |pl_orbpererr2|) / 2

### 최대 에러 (`_error`)
- **의미**: max(상위 불확실성, 하위 불확실성)
- **사용**: 항성 온도, 중력 (보수적 접근)
- **예시**: `st_teff_error` = max(|st_tefferr1|, |st_tefferr2|)

---

## 🎯 머신러닝 활용 팁

### 필수 Feature
- `pl_orbper` (궤도 주기)
- `pl_rade` (행성 반지름)
- `pl_eqt` (평형 온도)
- `pl_insol` (복사 플럭스)
- `pl_trandep` (통과 깊이) - ppm 단위
- `pl_trandurh` (통과 지속시간)

### 중요 Feature
- `st_teff` (항성 온도)
- `st_rad` (항성 반지름)
- `st_dist` (거리)
- `st_tmag` (TESS 등급)

### Limit Flag 처리
- `*_limit_flag != 0`인 데이터는 **제외** 권장
- TESS 데이터는 Kepler/K2보다 limit flag가 많은 편

### 정답 레이블 변환
TESS → Kepler 형식으로 통합 시:
```python
mapping = {
    'CP': 'CONFIRMED',
    'KP': 'CONFIRMED', 
    'FP': 'FALSE POSITIVE',
    'PC': 'CANDIDATE'
}
df['disposition'] = df['tfopwg_disp'].map(mapping)
```

### 데이터 품질
- TESS는 Kepler/K2보다 **컬럼 수가 적음**
- 많은 파라미터가 누락되어 있음 (이심률, 경사각, 항성 질량 등)
- **기본적인 통과 파라미터**와 **항성 기본 정보**만 포함

---

## 📊 Kepler/K2와의 차이점

| 항목 | Kepler/K2 | TESS |
|------|-----------|------|
| 컬럼 수 | 많음 (100~200개) | 적음 (57개) |
| 정답 형식 | CONFIRMED / FALSE POSITIVE / CANDIDATE | CP / FP / KP / PC |
| 통과 깊이 단위 | Kepler: ppm, K2: % | ppm |
| 이심률 | ✅ 있음 | ❌ 없음 |
| 궤도 경사각 | ✅ 있음 | ❌ 없음 |
| 항성 질량 | ✅ 있음 | ❌ 없음 |
| 항성 나이 | ✅ 있음 | ❌ 없음 |
| Limit Flag | K2만 있음 | ✅ 있음 |
| Symmetric Error Flag | ❌ 없음 | ✅ 있음 |

---

## 🔄 데이터 통합 시 주의사항

### 정답 레이블 통일
```python
# TESS → Kepler 형식
tess_to_kepler = {
    'CP': 'CONFIRMED',
    'KP': 'CONFIRMED',
    'FP': 'FALSE POSITIVE',
    'PC': 'CANDIDATE'
}
```

### 누락 컬럼 처리
- TESS에 없는 컬럼은 `NaN`으로 채우기
- 예: `pl_orbeccen`, `pl_orbincl`, `st_mass`, `st_age` 등

### 컬럼명 통일
- TESS → Kepler 형식으로 변환
  - `pl_trandurh` → `koi_duration`
  - `st_teff` → `koi_steff`
  - `st_rad` → `koi_srad`

---

## 💾 통합 데이터셋 생성 예제

```python
import pandas as pd

# TESS 데이터 로드
df_tess = pd.read_csv('tess_merged.csv')

# 정답 레이블 변환
label_mapping = {
    'CP': 'CONFIRMED',
    'KP': 'CONFIRMED',
    'FP': 'FALSE POSITIVE',
    'PC': 'CANDIDATE'
}
df_tess['disposition'] = df_tess['tfopwg_disp'].map(label_mapping)

# 컬럼명 변환 (Kepler 형식)
column_rename = {
    'pl_orbper': 'koi_period',
    'pl_rade': 'koi_prad',
    'pl_eqt': 'koi_teq',
    'pl_insol': 'koi_insol',
    'pl_trandep': 'koi_depth',  # 이미 ppm 단위
    'pl_trandurh': 'koi_duration',
    'st_teff': 'koi_steff',
    'st_rad': 'koi_srad',
    'st_logg': 'koi_slogg',
    'ra': 'ra',
    'dec': 'dec'
}
df_tess = df_tess.rename(columns=column_rename)

# 누락 컬럼 추가 (NaN)
missing_cols = ['koi_eccen', 'koi_sma', 'koi_incl', 'koi_smass', 'koi_sage']
for col in missing_cols:
    df_tess[col] = np.nan

# Kepler 데이터와 통합
df_kepler = pd.read_csv('kepler_merged.csv')
df_combined = pd.concat([df_kepler, df_tess], ignore_index=True)
```

---

생성일: 2025년 10월 5일
