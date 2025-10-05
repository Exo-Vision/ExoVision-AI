# K2 Merged 데이터셋 컬럼 설명서

**데이터 크기**: 4,004 행 × 201 컬럼  
**원본 컬럼**: 295개 → **통합 후**: 201개 (94개 컬럼 축소)

---

## 📌 기본 정보

| 컬럼명 | 설명 | 단위 | 타입 |
|--------|------|------|------|
| `rowid` | 행 ID | - | 정수 |
| `pl_name` | 행성 이름 | - | 문자열 |
| `hostname` | 항성 이름 | - | 문자열 |
| `pl_letter` | 행성 문자 (b, c, d...) | - | 문자 |
| `k2_name` | K2 ID | - | 문자열 |
| `epic_hostname` | EPIC HOST ID | - | 문자열 |
| `epic_candname` | EPIC CANDIDATE ID | - | 문자열 |
| `hd_name` | HD ID | - | 문자열 |
| `hip_name` | HIP ID | - | 문자열 |
| `tic_id` | TIC ID | - | 문자열 |
| `gaia_id` | GAIA ID | - | 문자열 |
| `default_flag` | 기본 파라미터 세트 플래그 | - | 0 / 1 |

---

## 🎯 정답 (Target)

| 컬럼명 | 설명 | 단위 | 값 | 비고 |
|--------|------|------|-----|------|
| **`disposition`** | **외계행성 판별 결과** (정답) | - | CONFIRMED / FALSE POSITIVE / CANDIDATE | **머신러닝 타겟** |
| `disp_refname` | 판별 참고 문헌 | - | 문자열 | - |

---

## 🔍 발견 정보

| 컬럼명 | 설명 | 값 |
|--------|------|-----|
| `discoverymethod` | 발견 방법 | Transit / Radial Velocity 등 |
| `disc_year` | 발견 연도 | 년도 |
| `disc_refname` | 발견 참고 문헌 | 문자열 |
| `disc_pubdate` | 발견 발표 날짜 | 날짜 |
| `disc_locale` | 발견 장소 | Space / Ground |
| `disc_facility` | 발견 시설 | K2 / Kepler 등 |
| `disc_telescope` | 망원경 이름 | 문자열 |
| `disc_instrument` | 관측 장비 | 문자열 |

---

## 🌍 궤도 파라미터

| 컬럼명 | 설명 | 단위 | 에러 컬럼 | 통합 전략 | Limit Flag |
|--------|------|------|----------|-----------|------------|
| **`pl_orbper`** | **궤도 주기** | days | `pl_orbper_error` | 평균 | - |
| `pl_orbsmax` | 궤도 장반경 | au | `pl_orbsmax_error` | 평균 | `pl_orbsmax_limit_flag` |
| **`pl_orbeccen`** | **궤도 이심률** | 무차원 | `pl_orbeccen_error_upper`<br>`pl_orbeccen_error_lower` | 분리 보존 | `pl_orbeccen_limit_flag` |
| **`pl_orbincl`** | **궤도 경사각** | degrees | `pl_orbincl_error` | 평균 | `pl_orbincl_limit_flag` |
| `pl_tranmid` | 통과 중심 시각 | BJD | `pl_tranmid_error` | 평균 | `pl_tranmid_limit_flag` |
| `pl_imppar` | 충돌 파라미터 | 무차원 | `pl_imppar_error` | 평균 | `pl_imppar_limit_flag` |
| `pl_orbtper` | 근점 통과 시각 | days | `pl_orbtper_error` | 평균 | `pl_orbtper_limit_flag` |
| `pl_orblper` | 근점 인수 | degrees | `pl_orblper_error` | 평균 | `pl_orblper_limit_flag` |

---

## 🪐 행성 물리량 - 반지름

| 컬럼명 | 설명 | 단위 | 에러 컬럼 | 통합 전략 | Limit Flag |
|--------|------|------|----------|-----------|------------|
| **`pl_rade`** | **행성 반지름** (지구 단위) | Earth radii | `pl_rade_error` | 평균 | `pl_rade_limit_flag` |
| `pl_radj` | 행성 반지름 (목성 단위) | Jupiter radii | `pl_radj_error` | 평균 | `pl_radj_limit_flag` |

---

## 🪐 행성 물리량 - 질량 ⚠️

> **중요**: 질량 컬럼은 `_limit_flag`를 반드시 확인해야 합니다!  
> - `_limit_flag = 0`: 정상 측정값
> - `_limit_flag = 1`: 상한값 (실제 값은 이보다 작음)
> - `_limit_flag = -1`: 하한값 (실제 값은 이보다 큼)

| 컬럼명 | 설명 | 단위 | 에러 컬럼 | Limit Flag | 비고 |
|--------|------|------|----------|------------|------|
| `pl_masse` | 행성 질량 (지구) | Earth mass | `pl_masse_error` | `pl_masse_limit_flag` | 직접 측정 |
| `pl_massj` | 행성 질량 (목성) | Jupiter mass | `pl_massj_error` | `pl_massj_limit_flag` | 직접 측정 |
| `pl_msinie` | Mass×sin(i) (지구) | Earth mass | `pl_msinie_error` | `pl_msinie_limit_flag` | 시선속도 |
| `pl_msinij` | Mass×sin(i) (목성) | Jupiter mass | `pl_msinij_error` | `pl_msinij_limit_flag` | 시선속도 |
| `pl_cmasse` | 계산 질량 (지구) | Earth mass | `pl_cmasse_error` | `pl_cmasse_limit_flag` | 간접 추정 |
| `pl_cmassj` | 계산 질량 (목성) | Jupiter mass | `pl_cmassj_error` | `pl_cmassj_limit_flag` | 간접 추정 |
| **`pl_bmasse`** | **최적 질량 (지구)** | Earth mass | `pl_bmasse_error` | `pl_bmasse_limit_flag` | **권장** |
| **`pl_bmassj`** | **최적 질량 (목성)** | Jupiter mass | `pl_bmassj_error` | `pl_bmassj_limit_flag` | **권장** |
| `pl_bmassprov` | 최적 질량 출처 | - | - | - | Mass / Msini |

---

## 🪐 행성 물리량 - 기타

| 컬럼명 | 설명 | 단위 | 에러 컬럼 | 통합 전략 | Limit Flag |
|--------|------|------|----------|-----------|------------|
| `pl_dens` | 행성 밀도 | g/cm³ | `pl_dens_error` | 평균 | `pl_dens_limit_flag` |
| **`pl_insol`** | **복사 플럭스** | Earth flux | `pl_insol_error` | 평균 | `pl_insol_limit_flag` |
| **`pl_eqt`** | **평형 온도** | Kelvin | `pl_eqt_error` | 평균 | `pl_eqt_limit_flag` |

---

## 🌟 통과(Transit) 파라미터

| 컬럼명 | 설명 | 단위 | 에러 컬럼 | 통합 전략 | Limit Flag |
|--------|------|------|----------|-----------|------------|
| **`pl_trandep`** | **통과 깊이** | **%** (주의!) | `pl_trandep_error` | 평균 | `pl_trandep_limit_flag` |
| **`pl_trandur`** | **통과 지속시간** | hours | `pl_trandur_error` | 평균 | `pl_trandur_limit_flag` |
| `pl_ratdor` | 거리/반지름 비율 | 무차원 | `pl_ratdor_error` | 평균 | `pl_ratdor_limit_flag` |
| `pl_ratror` | 행성/항성 반지름 비율 | 무차원 | `pl_ratror_error` | 평균 | `pl_ratror_limit_flag` |
| `pl_occdep` | 가림 깊이 | % | `pl_occdep_error` | 평균 | `pl_occdep_limit_flag` |

> ⚠️ **중요**: K2의 `pl_trandep`는 **%** 단위입니다! (Kepler는 ppm)  
> - Kepler로 통합 시 단위 변환 필요: `% × 10,000 = ppm`

---

## 📡 시선속도 파라미터

| 컬럼명 | 설명 | 단위 | 에러 컬럼 | 통합 전략 | Limit Flag |
|--------|------|------|----------|-----------|------------|
| `pl_rvamp` | 시선속도 진폭 | m/s | `pl_rvamp_error` | 평균 | `pl_rvamp_limit_flag` |
| `pl_projobliq` | 투영 경사각 | degrees | `pl_projobliq_error` | 평균 | `pl_projobliq_limit_flag` |
| `pl_trueobliq` | 실제 경사각 | degrees | `pl_trueobliq_error` | 평균 | `pl_trueobliq_limit_flag` |

---

## ⭐ 항성 물리량

| 컬럼명 | 설명 | 단위 | 에러 컬럼 | 통합 전략 | Limit Flag |
|--------|------|------|----------|-----------|------------|
| **`st_teff`** | **항성 유효 온도** | Kelvin | `st_teff_error` | 최대 (보수적) | `st_teff_limit_flag` |
| **`st_rad`** | **항성 반지름** | Solar radii | `st_rad_error` | 평균 | `st_rad_limit_flag` |
| **`st_mass`** | **항성 질량** | Solar mass | `st_mass_error` | 최대 (보수적) | `st_mass_limit_flag` |
| `st_met` | 항성 금속 함량 | dex | `st_met_error` | 평균 | `st_met_limit_flag` |
| `st_metratio` | 금속 함량 비율 | - | - | - | - |
| `st_lum` | 항성 광도 | log(Solar) | `st_lum_error` | 평균 | `st_lum_limit_flag` |
| **`st_logg`** | **항성 표면 중력** | log10(cm/s²) | `st_logg_error` | 최대 (보수적) | `st_logg_limit_flag` |
| `st_age` | 항성 나이 | Gyr | `st_age_error` | 평균 | `st_age_limit_flag` |
| `st_dens` | 항성 밀도 | g/cm³ | `st_dens_error` | 평균 | `st_dens_limit_flag` |
| `st_vsin` | 항성 회전속도 | km/s | `st_vsin_error` | 평균 | `st_vsin_limit_flag` |
| `st_rotp` | 항성 회전주기 | days | `st_rotp_error` | 평균 | `st_rotp_limit_flag` |
| `st_radv` | 시선속도 | km/s | `st_radv_error` | 평균 | `st_radv_limit_flag` |
| `st_spectype` | 스펙트럼 타입 | - | - | - | - |

---

## 📍 위치 및 운동

| 컬럼명 | 설명 | 단위 | 에러 컬럼 |
|--------|------|------|----------|
| `rastr` | 적경 (육십분법) | sexagesimal | - |
| **`ra`** | **적경** | degrees | - |
| `decstr` | 적위 (육십분법) | sexagesimal | - |
| **`dec`** | **적위** | degrees | - |
| `glat` | 은하 위도 | degrees | - |
| `glon` | 은하 경도 | degrees | - |
| `elat` | 황도 위도 | degrees | - |
| `elon` | 황도 경도 | degrees | - |

### 고유운동

| 컬럼명 | 설명 | 단위 | 에러 컬럼 |
|--------|------|------|----------|
| `sy_pm` | 총 고유운동 | mas/yr | `sy_pm_error` |
| `sy_pmra` | 고유운동 (RA) | mas/yr | `sy_pmra_error` |
| `sy_pmdec` | 고유운동 (Dec) | mas/yr | `sy_pmdec_error` |

### 거리 및 시차

| 컬럼명 | 설명 | 단위 | 에러 컬럼 |
|--------|------|------|----------|
| **`sy_dist`** | **거리** | pc | `sy_dist_error` |
| `sy_plx` | 시차 | mas | `sy_plx_error` |

---

## 🔭 측광 등급

### 주요 밴드

| 컬럼명 | 설명 | 단위 | 에러 컬럼 |
|--------|------|------|----------|
| `sy_bmag` | B (Johnson) 등급 | magnitude | `sy_bmag_error` |
| `sy_vmag` | V (Johnson) 등급 | magnitude | `sy_vmag_error` |
| `sy_jmag` | J (2MASS) 등급 | magnitude | `sy_jmag_error` |
| `sy_hmag` | H (2MASS) 등급 | magnitude | `sy_hmag_error` |
| `sy_kmag` | Ks (2MASS) 등급 | magnitude | `sy_kmag_error` |

### Sloan 밴드

| 컬럼명 | 설명 | 단위 | 에러 컬럼 |
|--------|------|------|----------|
| `sy_umag` | u (Sloan) 등급 | magnitude | `sy_umag_error` |
| `sy_gmag` | g (Sloan) 등급 | magnitude | `sy_gmag_error` |
| `sy_rmag` | r (Sloan) 등급 | magnitude | `sy_rmag_error` |
| `sy_imag` | i (Sloan) 등급 | magnitude | `sy_imag_error` |
| `sy_zmag` | z (Sloan) 등급 | magnitude | `sy_zmag_error` |

### WISE 밴드

| 컬럼명 | 설명 | 단위 | 에러 컬럼 |
|--------|------|------|----------|
| `sy_w1mag` | W1 (WISE) 등급 | magnitude | `sy_w1mag_error` |
| `sy_w2mag` | W2 (WISE) 등급 | magnitude | `sy_w2mag_error` |
| `sy_w3mag` | W3 (WISE) 등급 | magnitude | `sy_w3mag_error` |
| `sy_w4mag` | W4 (WISE) 등급 | magnitude | `sy_w4mag_error` |

### 기타

| 컬럼명 | 설명 | 단위 | 에러 컬럼 |
|--------|------|------|----------|
| `sy_gaiamag` | Gaia 등급 | magnitude | `sy_gaiamag_error` |
| `sy_icmag` | I (Cousins) 등급 | magnitude | `sy_icmag_error` |
| `sy_tmag` | TESS 등급 | magnitude | `sy_tmag_error` |
| `sy_kepmag` | Kepler 등급 | magnitude | `sy_kepmag_error` |

---

## 🚩 발견 방법 플래그

| 컬럼명 | 설명 | 값 |
|--------|------|-----|
| `rv_flag` | 시선속도로 발견 | 0 / 1 |
| `pul_flag` | 펄서 타이밍으로 발견 | 0 / 1 |
| `ptv_flag` | 맥동 타이밍으로 발견 | 0 / 1 |
| `tran_flag` | 통과로 발견 | 0 / 1 |
| `ast_flag` | 위치측정으로 발견 | 0 / 1 |
| `obm_flag` | 궤도 밝기 변조로 발견 | 0 / 1 |
| `micro_flag` | 미세중력렌즈로 발견 | 0 / 1 |
| `etv_flag` | 식 타이밍 변화로 발견 | 0 / 1 |
| `ima_flag` | 직접 촬영으로 발견 | 0 / 1 |
| `dkin_flag` | 원반 운동학으로 발견 | 0 / 1 |

---

## 📝 시스템 정보

| 컬럼명 | 설명 |
|--------|------|
| `sy_snum` | 항성 개수 |
| `sy_pnum` | 행성 개수 |
| `sy_mnum` | 위성 개수 |
| `cb_flag` | 쌍성 주위 궤도 플래그 |
| `soltype` | 해법 타입 |
| `pl_controv_flag` | 논란 플래그 |
| `ttv_flag` | 통과 타이밍 변화 플래그 |

---

## 📚 참고 문헌

| 컬럼명 | 설명 |
|--------|------|
| `pl_refname` | 행성 파라미터 참고 문헌 |
| `st_refname` | 항성 파라미터 참고 문헌 |
| `sy_refname` | 시스템 파라미터 참고 문헌 |

---

## 📅 메타데이터

| 컬럼명 | 설명 |
|--------|------|
| `rowupdate` | 마지막 업데이트 날짜 |
| `pl_pubdate` | 행성 파라미터 발표 날짜 |
| `releasedate` | 릴리스 날짜 |
| `pl_nnotes` | 노트 개수 |
| `k2_campaigns` | K2 캠페인 |
| `k2_campaigns_num` | K2 캠페인 개수 |
| `st_nphot` | 측광 시계열 개수 |
| `st_nrvc` | 시선속도 시계열 개수 |
| `st_nspec` | 스펙트럼 측정 개수 |
| `pl_nespec` | 식 스펙트럼 개수 |
| `pl_ntranspec` | 통과 스펙트럼 개수 |
| `pl_ndispec` | 직접 촬영 스펙트럼 개수 |

---

## 💡 머신러닝 활용 팁

### 필수 Feature
- `pl_orbper` (궤도 주기)
- `pl_rade` (행성 반지름)
- `pl_eqt` (평형 온도)
- `pl_insol` (복사 플럭스)
- `pl_trandep` (통과 깊이) - ⚠️ **% 단위 주의!**
- `pl_trandur` (통과 지속시간)

### Limit Flag 처리
- `*_limit_flag != 0`인 데이터는 **제외** 권장
- 특히 질량 관련 컬럼 (`pl_masse`, `pl_massj` 등)

### 데이터 품질
- `default_flag = 1`인 행만 사용 권장 (최적 파라미터)
- 발견 방법 플래그로 필터링 가능 (`tran_flag = 1` 등)

---

생성일: 2025년 10월 5일
