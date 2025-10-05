# 세 데이터셋(Kepler, K2, TESS) 공통 컬럼 분석 보고서

## 📊 요약 통계

- **전체 공통 컬럼 수**: 26개
- **세 데이터셋 모두 존재**: 13개
- **Kepler & K2만 존재**: 13개  
- **Kepler & TESS만 존재**: 0개
- **단위 변환 필요**: 1개

---

## ⚠️ 중요: 단위 변환이 필요한 컬럼

### 통과 깊이 (Transit Depth)

**단위 불일치:**
- Kepler: `ppm` (parts per million)
- TESS: `ppm` (parts per million)  
- K2: `%` (퍼센트)

**변환 공식:**
```
K2 값 (ppm) = K2 값 (%) × 10,000
```

**예시:**
- K2에서 0.5% → 5,000 ppm
- K2에서 1.2% → 12,000 ppm

---

## 📋 카테고리별 공통 컬럼 상세 정보

### 1️⃣ 정답 (외계행성 판별 결과)

| 컬럼 설명 | Kepler | TESS | K2 | 단위 | 비고 |
|----------|--------|------|----|----|------|
| 외계행성 판별 결과 | `koi_disposition` | `tfopwg_disp` | `disposition` | categorical | Kepler/K2: CONFIRMED/FALSE POSITIVE/CANDIDATE<br>TESS: CP/FP/KP/PC |

---

### 2️⃣ 궤도 파라미터

| 컬럼 설명 | Kepler | TESS | K2 | 단위 | 단위 일치 |
|----------|--------|------|----|----|---------|
| 궤도 주기 | `koi_period` | `pl_orbper` | `pl_orbper` | days | ✓ |
| 통과 중심 시각 | `koi_time0` | `pl_tranmid` | `pl_tranmid` | BJD | ✓ |
| 궤도 이심률 | `koi_eccen` | ❌ | `pl_orbeccen` | 무차원 | ✓ |
| 궤도 장반경 | `koi_sma` | ❌ | `pl_orbsmax` | au | ✓ |
| 궤도 경사각 | `koi_incl` | ❌ | `pl_orbincl` | degrees | ✓ |
| 충돌 파라미터 | `koi_impact` | ❌ | `pl_imppar` | 무차원 | ✓ |
| 거리/반지름 비율 | `koi_dor` | ❌ | `pl_ratdor` | 무차원 | ✓ |

---

### 3️⃣ 통과(Transit) 파라미터

| 컬럼 설명 | Kepler | TESS | K2 | 단위 | 단위 일치 |
|----------|--------|------|----|----|---------|
| 통과 지속시간 | `koi_duration` | `pl_trandurh` | `pl_trandur` | hours | ✓ |
| **통과 깊이** | `koi_depth` | `pl_trandep` | `pl_trandep` | Kepler/TESS: ppm<br>K2: % | **❌ 변환 필요** |
| 반지름 비율 | `koi_ror` | ❌ | `pl_ratror` | 무차원 | ✓ |

---

### 4️⃣ 행성 물리량

| 컬럼 설명 | Kepler | TESS | K2 | 단위 | 단위 일치 |
|----------|--------|------|----|----|---------|
| 행성 반지름 | `koi_prad` | `pl_rade` | `pl_rade` | Earth radii | ✓ |
| 평형 온도 | `koi_teq` | `pl_eqt` | `pl_eqt` | Kelvin | ✓ |
| 복사 플럭스 | `koi_insol` | `pl_insol` | `pl_insol` | Earth flux | ✓ |

---

### 5️⃣ 항성 물리량

| 컬럼 설명 | Kepler | TESS | K2 | 단위 | 단위 일치 |
|----------|--------|------|----|----|---------|
| 항성 유효 온도 | `koi_steff` | `st_teff` | `st_teff` | Kelvin | ✓ |
| 항성 표면 중력 | `koi_slogg` | `st_logg` | `st_logg` | log10(cm/s²) | ✓ |
| 항성 금속 함량 | `koi_smet` | ❌ | `st_met` | dex | ✓ |
| 항성 반지름 | `koi_srad` | `st_rad` | `st_rad` | Solar radii | ✓ |
| 항성 질량 | `koi_smass` | ❌ | `st_mass` | Solar mass | ✓ |
| 항성 나이 | `koi_sage` | ❌ | `st_age` | Gyr | ✓ |
| 항성 밀도 | `koi_srho` | ❌ | `st_dens` | g/cm³ | ✓ |

---

### 6️⃣ 위치 정보

| 컬럼 설명 | Kepler | TESS | K2 | 단위 | 단위 일치 |
|----------|--------|------|----|----|---------|
| 적경 | `ra` | `ra` | `ra` | degrees | ✓ |
| 적위 | `dec` | `dec` | `dec` | degrees | ✓ |

---

### 7️⃣ 측광 등급

| 컬럼 설명 | Kepler | TESS | K2 | 단위 | 단위 일치 |
|----------|--------|------|----|----|---------|
| J-band 등급 | `koi_jmag` | ❌ | `sy_jmag` | magnitude | ✓ |
| H-band 등급 | `koi_hmag` | ❌ | `sy_hmag` | magnitude | ✓ |
| K-band 등급 | `koi_kmag` | ❌ | `sy_kmag` | magnitude | ✓ |

---

## 🔍 데이터셋별 특징

### Kepler 데이터셋
- 가장 많은 고유 컬럼 보유
- 모든 기본적인 행성 및 항성 파라미터 포함
- 다른 데이터셋 통합의 기준이 됨

### TESS 데이터셋  
- 가장 적은 컬럼 수
- 기본적인 통과(transit) 및 항성 파라미터만 포함
- 많은 고급 파라미터 누락 (이심률, 궤도 장반경, 항성 질량 등)

### K2 데이터셋
- Kepler와 가장 유사한 구조
- 대부분의 Kepler 컬럼과 매핑 가능
- **주의**: 통과 깊이만 단위가 다름 (% vs ppm)

---

## 💡 데이터 통합 시 주의사항

1. **단위 변환**: K2의 통과 깊이(`pl_trandep`)를 반드시 % → ppm으로 변환
2. **결측치 처리**: TESS 데이터는 많은 컬럼이 누락되어 있음
3. **정답 레이블 통일**: 
   - Kepler/K2: `CONFIRMED`, `FALSE POSITIVE`, `CANDIDATE`
   - TESS: `CP` (Confirmed Planet), `FP` (False Positive), `KP` (Known Planet), `PC` (Planet Candidate)
4. **컬럼명 통일**: 통합 시 Kepler 컬럼명 기준으로 통일 권장

---

## 📝 통합 데이터셋 생성 시 권장 단계

1. **Kepler 데이터를 기준으로 설정**
2. **K2 데이터 매핑**:
   - 컬럼명 변경 (K2 → Kepler)
   - 통과 깊이 단위 변환 (% × 10,000)
3. **TESS 데이터 매핑**:
   - 컬럼명 변경 (TESS → Kepler)
   - 누락된 컬럼은 NaN으로 처리
4. **정답 레이블 통일**:
   - TESS의 CP, KP → CONFIRMED
   - TESS의 FP → FALSE POSITIVE  
   - TESS의 PC → CANDIDATE
5. **데이터 결합**: `pd.concat()` 또는 유사 방법 사용

---

생성일: 2025년 10월 5일
