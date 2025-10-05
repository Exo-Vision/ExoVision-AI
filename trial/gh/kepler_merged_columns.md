# 통합 데이터셋 컬럼 설명서

## Kepler Merged 데이터셋 (`kepler_merged.csv`)

**데이터 크기**: 9,564 행 × 119 컬럼  
**원본 컬럼**: 141개 → **통합 후**: 119개 (22개 컬럼 축소)

---

### 📌 기본 정보

| 컬럼명 | 설명 | 단위 | 타입 |
|--------|------|------|------|
| `rowid` | 행 ID | - | 정수 |
| `kepid` | Kepler ID | - | 정수 |
| `kepoi_name` | KOI 이름 | - | 문자열 |
| `kepler_name` | Kepler 행성 이름 | - | 문자열 |

---

### 🎯 정답 (Target)

| 컬럼명 | 설명 | 단위 | 값 | 비고 |
|--------|------|------|-----|------|
| **`koi_disposition`** | **외계행성 판별 결과** (정답) | - | CONFIRMED / FALSE POSITIVE / CANDIDATE | **머신러닝 타겟** |
| `koi_pdisposition` | Kepler 데이터만 사용한 판별 | - | CANDIDATE / FALSE POSITIVE | 참고용 |
| `koi_score` | 판별 점수 | - | 0.0 ~ 1.0 | 1.0에 가까울수록 행성 |

---

### 🔧 품질 플래그

| 컬럼명 | 설명 | 값 |
|--------|------|-----|
| `koi_fpflag_nt` | Not Transit-Like 거짓 양성 플래그 | 0 / 1 |
| `koi_fpflag_ss` | Stellar Eclipse 거짓 양성 플래그 | 0 / 1 |
| `koi_fpflag_co` | Centroid Offset 거짓 양성 플래그 | 0 / 1 |
| `koi_fpflag_ec` | Ephemeris Match 거짓 양성 플래그 | 0 / 1 |

---

### 🌍 궤도 파라미터

| 컬럼명 | 설명 | 단위 | 에러 컬럼 | 통합 전략 |
|--------|------|------|----------|-----------|
| **`koi_period`** | **궤도 주기** | days | `koi_period_error` | 평균 |
| `koi_time0bk` | 통과 시각 (BKJD) | BKJD | `koi_time0bk_error` | 평균 |
| `koi_time0` | 통과 시각 (BJD) | BJD | `koi_time0_error` | 평균 |
| **`koi_eccen`** | **궤도 이심률** | 무차원 | `koi_eccen_error_upper`<br>`koi_eccen_error_lower` | 분리 보존 |
| `koi_longp` | 근점 경도 | degrees | `koi_longp_error` | 평균 |
| `koi_impact` | 충돌 파라미터 | 무차원 | `koi_impact_error` | 평균 |
| `koi_sma` | 궤도 장반경 | au | `koi_sma_error` | 평균 |
| **`koi_incl`** | **궤도 경사각** | degrees | `koi_incl_error` | 평균 |
| `koi_dor` | 행성-항성 거리/항성 반지름 비율 | 무차원 | `koi_dor_error` | 평균 |

---

### 🌟 통과(Transit) 파라미터

| 컬럼명 | 설명 | 단위 | 에러 컬럼 | 통합 전략 |
|--------|------|------|----------|-----------|
| **`koi_duration`** | **통과 지속시간** | hours | `koi_duration_error` | 평균 |
| `koi_ingress` | 진입 지속시간 | hours | `koi_ingress_error` | 평균 |
| **`koi_depth`** | **통과 깊이** | ppm | `koi_depth_error` | 평균 |
| `koi_ror` | 행성-항성 반지름 비율 | 무차원 | `koi_ror_error` | 평균 |

---

### 🪐 행성 물리량

| 컬럼명 | 설명 | 단위 | 에러 컬럼 | 통합 전략 |
|--------|------|------|----------|-----------|
| **`koi_prad`** | **행성 반지름** | Earth radii | `koi_prad_error` | 평균 |
| **`koi_teq`** | **평형 온도** | Kelvin | `koi_teq_error` | 평균 |
| **`koi_insol`** | **복사 플럭스** | Earth flux | `koi_insol_error` | 평균 |

---

### ⭐ 항성 물리량

| 컬럼명 | 설명 | 단위 | 에러 컬럼 | 통합 전략 |
|--------|------|------|----------|-----------|
| **`koi_steff`** | **항성 유효 온도** | Kelvin | `koi_steff_error` | 최대 (보수적) |
| **`koi_slogg`** | **항성 표면 중력** | log10(cm/s²) | `koi_slogg_error` | 최대 (보수적) |
| `koi_smet` | 항성 금속 함량 | dex | `koi_smet_error` | 평균 |
| **`koi_srad`** | **항성 반지름** | Solar radii | `koi_srad_error` | 평균 |
| **`koi_smass`** | **항성 질량** | Solar mass | `koi_smass_error` | 최대 (보수적) |
| `koi_sage` | 항성 나이 | Gyr | `koi_sage_error` | 평균 |
| `koi_srho` | 항성 밀도 | g/cm³ | `koi_srho_error` | 최대 (보수적) |

---

### 📍 위치 정보

| 컬럼명 | 설명 | 단위 |
|--------|------|------|
| **`ra`** | **적경** | degrees |
| **`dec`** | **적위** | degrees |

---

### 🔭 측광 등급

| 컬럼명 | 설명 | 단위 |
|--------|------|------|
| `koi_kepmag` | Kepler-band 등급 | magnitude |
| `koi_gmag` | g'-band 등급 | magnitude |
| `koi_rmag` | r'-band 등급 | magnitude |
| `koi_imag` | i'-band 등급 | magnitude |
| `koi_zmag` | z'-band 등급 | magnitude |
| `koi_jmag` | J-band 등급 | magnitude |
| `koi_hmag` | H-band 등급 | magnitude |
| `koi_kmag` | K-band 등급 | magnitude |

---

### 📊 통계 정보

| 컬럼명 | 설명 | 단위 |
|--------|------|------|
| `koi_model_snr` | 통과 신호 대 잡음 비율 | - |
| `koi_count` | 행성 개수 | 개 |
| `koi_num_transits` | 관측된 통과 횟수 | 회 |
| `koi_max_sngle_ev` | 최대 단일 이벤트 통계 | - |
| `koi_max_mult_ev` | 최대 다중 이벤트 통계 | - |

---

### 📝 메타데이터

| 컬럼명 | 설명 |
|--------|------|
| `koi_vet_stat` | 검증 상태 |
| `koi_vet_date` | 마지막 업데이트 날짜 |
| `koi_disp_prov` | 판별 출처 |
| `koi_comment` | 코멘트 |
| `koi_fittype` | 행성 피팅 타입 |
| `koi_limbdark_mod` | Limb Darkening 모델 |
| `koi_parm_prov` | 파라미터 출처 |
| `koi_sparprov` | 항성 파라미터 출처 |
| `koi_trans_mod` | 통과 모델 |
| `koi_tce_delivname` | TCE Delivery |
| `koi_quarters` | 관측 Quarter |

---

## 💡 에러 컬럼 통합 규칙 요약

### 평균 에러 (`_error`)
- **의미**: (상위 불확실성 + 하위 불확실성) / 2
- **사용**: 궤도 주기, 반지름, 온도 등 대부분의 측정값
- **예시**: `koi_period_error` = (|koi_period_err1| + |koi_period_err2|) / 2

### 최대 에러 (`_error`)
- **의미**: max(상위 불확실성, 하위 불확실성)
- **사용**: 항성 질량, 온도, 중력 등 보수적 접근 필요
- **예시**: `koi_steff_error` = max(|koi_steff_err1|, |koi_steff_err2|)

### 분리 보존 (`_error_upper`, `_error_lower`)
- **의미**: 비대칭 에러를 그대로 유지
- **사용**: 궤도 이심률 (경계값 0 근처 비대칭)
- **예시**: 
  - `koi_eccen_error_upper` = |koi_eccen_err1|
  - `koi_eccen_error_lower` = |koi_eccen_err2|

---

## 🎯 머신러닝 활용 팁

### 필수 Feature
- `koi_period` (궤도 주기)
- `koi_prad` (행성 반지름)
- `koi_teq` (평형 온도)
- `koi_insol` (복사 플럭스)
- `koi_depth` (통과 깊이)
- `koi_duration` (통과 지속시간)

### 중요 Feature
- `koi_steff` (항성 온도)
- `koi_srad` (항성 반지름)
- `koi_model_snr` (신호 대 잡음 비율)
- `koi_impact` (충돌 파라미터)

### 품질 필터링
- `koi_fpflag_*` 플래그들을 확인하여 품질 낮은 데이터 제외
- `koi_score`가 너무 낮은 데이터 제외 고려

---

생성일: 2025년 10월 5일
