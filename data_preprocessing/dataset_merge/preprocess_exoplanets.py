"""
외계행성 데이터셋 결측치 처리 및 최종 데이터셋 생성
- 물리 법칙 기반 계산
- 머신러닝 회귀 모델
- 통계적 추정
- 불필요 컬럼 드랍
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 물리 상수
G = 6.67430e-11  # 중력 상수 (m^3 kg^-1 s^-2)
R_sun = 6.96e8   # 태양 반지름 (m)
M_sun = 1.989e30 # 태양 질량 (kg)
AU = 1.496e11    # 천문단위 (m)
STEFAN_BOLTZMANN = 5.67e-8  # 슈테판-볼츠만 상수


def calculate_stellar_mass(df, rad_col='koi_srad', logg_col='koi_slogg'):
    """항성 질량 계산: M = R² × 10^(logg - 4.44)"""
    mask = df[rad_col].notna() & df[logg_col].notna()
    
    if mask.sum() > 0:
        R = df.loc[mask, rad_col]
        logg = df.loc[mask, logg_col]
        
        # M = R² × 10^(logg - logg_sun)
        M = R**2 * 10**(logg - 4.44)
        
        # 이상치 필터링 (0.1 ~ 10 M_sun)
        valid = (M >= 0.1) & (M <= 10.0)
        
        return M[valid], mask
    
    return pd.Series(dtype=float), mask


def calculate_semimajor_axis(df, period_col='koi_period', mass_col='koi_smass'):
    """반장축 계산: a = (P² × M)^(1/3) [케플러 제3법칙]"""
    mask = df[period_col].notna() & df[mass_col].notna()
    
    if mask.sum() > 0:
        P_days = df.loc[mask, period_col]
        M_star = df.loc[mask, mass_col]
        
        # 주기를 년 단위로 변환
        P_years = P_days / 365.25
        
        # a³ = P² × M (a in AU, P in years, M in solar masses)
        a = (P_years**2 * M_star)**(1/3)
        
        # 이상치 필터링 (0.001 ~ 100 AU)
        valid = (a >= 0.001) & (a <= 100.0)
        
        return a[valid], mask
    
    return pd.Series(dtype=float), mask


def calculate_surface_gravity(df, mass_col='koi_smass', rad_col='koi_srad'):
    """표면중력 계산: logg = log10(G×M/R²) + 4.44"""
    mask = df[mass_col].notna() & df[rad_col].notna()
    
    if mask.sum() > 0:
        M = df.loc[mask, mass_col]
        R = df.loc[mask, rad_col]
        
        # logg = log10(g/g_sun) + logg_sun
        # g = G × M / R²
        logg = np.log10(M / R**2) + 4.44
        
        # 이상치 필터링 (2.0 ~ 5.5)
        valid = (logg >= 2.0) & (logg <= 5.5)
        
        return logg[valid], mask
    
    return pd.Series(dtype=float), mask


def calculate_impact_parameter(df, sma_col='koi_sma', rad_col='koi_srad', incl_col='koi_incl'):
    """충격 매개변수 계산: b = (a/R_star) × cos(i)"""
    mask = df[sma_col].notna() & df[rad_col].notna() & df[incl_col].notna()
    
    if mask.sum() > 0:
        a = df.loc[mask, sma_col]  # AU
        R_star = df.loc[mask, rad_col]  # R_sun
        incl = df.loc[mask, incl_col]  # degrees
        
        # AU를 R_sun으로 변환 (1 AU = 215 R_sun)
        a_in_r_sun = a * 215.032
        
        # b = (a/R_star) × cos(i)
        b = (a_in_r_sun / R_star) * np.cos(np.radians(incl))
        
        # 이상치 필터링 (0 ~ 2)
        valid = (b >= 0) & (b <= 2.0)
        
        return b[valid], mask
    
    return pd.Series(dtype=float), mask


def calculate_transit_depth(df, prad_col='koi_prad', srad_col='koi_srad'):
    """통과 깊이 계산: depth ≈ (R_planet/R_star)² × 10^6 (ppm)"""
    mask = df[prad_col].notna() & df[srad_col].notna()
    
    if mask.sum() > 0:
        R_planet = df.loc[mask, prad_col]  # R_earth
        R_star = df.loc[mask, srad_col]  # R_sun
        
        # R_earth를 R_sun으로 변환 (1 R_sun = 109 R_earth)
        R_planet_in_r_sun = R_planet / 109.0
        
        # depth = (R_planet/R_star)² × 10^6
        depth = (R_planet_in_r_sun / R_star)**2 * 1e6
        
        # 이상치 필터링 (1 ~ 100000 ppm)
        valid = (depth >= 1) & (depth <= 100000)
        
        return depth[valid], mask
    
    return pd.Series(dtype=float), mask


def calculate_insolation_flux(df, sma_col='koi_sma', srad_col='koi_srad', teff_col='koi_steff'):
    """입사 플럭스 계산: insol = L_star / (4π × a²)"""
    mask = df[sma_col].notna() & df[srad_col].notna() & df[teff_col].notna()
    
    if mask.sum() > 0:
        a = df.loc[mask, sma_col]  # AU
        R_star = df.loc[mask, srad_col]  # R_sun
        T_eff = df.loc[mask, teff_col]  # K
        
        # L_star = 4π × R_star² × σ × T_eff⁴ (태양 광도 단위)
        L_star = R_star**2 * (T_eff / 5778)**4
        
        # insol = L_star / a² (지구 입사 플럭스 단위)
        insol = L_star / a**2
        
        # 이상치 필터링 (0.01 ~ 10000)
        valid = (insol >= 0.01) & (insol <= 10000)
        
        return insol[valid], mask
    
    return pd.Series(dtype=float), mask


def calculate_equilibrium_temperature(df, steff_col='koi_steff', srad_col='koi_srad', sma_col='koi_sma'):
    """평형 온도 계산: T_eq = T_star × (R_star/(2a))^0.5"""
    mask = df[steff_col].notna() & df[srad_col].notna() & df[sma_col].notna()
    
    if mask.sum() > 0:
        T_star = df.loc[mask, steff_col]  # K
        R_star = df.loc[mask, srad_col]  # R_sun
        a = df.loc[mask, sma_col]  # AU
        
        # AU를 R_sun으로 변환
        a_in_r_sun = a * 215.032
        
        # T_eq = T_star × (R_star/(2a))^0.5 (albedo=0.3 가정)
        T_eq = T_star * (R_star / (2 * a_in_r_sun))**0.5
        
        # 이상치 필터링 (50 ~ 5000 K)
        valid = (T_eq >= 50) & (T_eq <= 5000)
        
        return T_eq[valid], mask
    
    return pd.Series(dtype=float), mask


def fill_with_regression(df, target_col, feature_cols, min_samples=100):
    """회귀 모델로 결측치 채우기"""
    
    # 학습 데이터: target이 있고 모든 feature가 있는 행
    train_mask = df[target_col].notna()
    for col in feature_cols:
        train_mask &= df[col].notna()
    
    # 예측 데이터: target이 없지만 모든 feature가 있는 행
    predict_mask = df[target_col].isna()
    for col in feature_cols:
        predict_mask &= df[col].notna()
    
    if train_mask.sum() < min_samples:
        print(f"  ⚠️ {target_col}: 학습 데이터 부족 ({train_mask.sum()} < {min_samples})")
        return df
    
    if predict_mask.sum() == 0:
        print(f"  ℹ️ {target_col}: 채울 데이터 없음")
        return df
    
    # 학습
    X_train = df.loc[train_mask, feature_cols]
    y_train = df.loc[train_mask, target_col]
    
    # 스케일링
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Random Forest 회귀
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_scaled, y_train)
    
    # 예측
    X_predict = df.loc[predict_mask, feature_cols]
    X_predict_scaled = scaler.transform(X_predict)
    y_predict = rf.predict(X_predict_scaled)
    
    # 채우기
    df.loc[predict_mask, target_col] = y_predict
    
    print(f"  ✅ {target_col}: {predict_mask.sum()}개 채움 "
          f"(R² score: {rf.score(X_train_scaled, y_train):.3f})")
    
    return df


def preprocess_exoplanet_data():
    """메인 전처리 함수"""
    
    print("="*100)
    print("🚀 외계행성 데이터셋 전처리 시작")
    print("="*100)
    
    # 1. 데이터 로드
    print("\n📂 1. 데이터 로드")
    print("-"*100)
    df = pd.read_csv('datasets/exoplanets_integrated.csv')
    print(f"원본 데이터: {len(df)} 행 × {len(df.columns)} 컬럼")
    
    original_missing = df.isna().sum().sum()
    print(f"총 결측치: {original_missing:,} ({original_missing/(len(df)*len(df.columns))*100:.2f}%)")
    
    # 2. 불필요 컬럼 드랍
    print("\n❌ 2. 불필요 컬럼 드랍")
    print("-"*100)
    
    drop_cols = []
    
    # koi_kepmag: TESS 전체 결측, 행성 판별과 무관
    if 'koi_kepmag' in df.columns:
        drop_cols.append('koi_kepmag')
        print(f"  • koi_kepmag 드랍: 관측 밝기 (TESS 전체 결측, 행성 판별 무관)")
    
    # koi_time0bk: 관측 타이밍, 행성 물리량과 무관
    if 'koi_time0bk' in df.columns:
        drop_cols.append('koi_time0bk')
        print(f"  • koi_time0bk 드랍: 첫 통과 시각 (관측 타이밍, 행성 물리량 무관)")
    
    # koi_ingress: koi_duration과 완전 중복 (r=1.00), 결측 63.8%
    if 'koi_ingress' in df.columns:
        drop_cols.append('koi_ingress')
        print(f"  • koi_ingress 드랍: koi_duration과 중복 (r=1.00), 결측 63.8%")
    
    df = df.drop(columns=drop_cols)
    print(f"\n드랍 후: {len(df)} 행 × {len(df.columns)} 컬럼")
    
    # 3. 물리 법칙 기반 결측치 채우기
    print("\n⚗️ 3. 물리 법칙 기반 결측치 채우기")
    print("-"*100)
    
    # 3.1 항성 질량 (koi_smass)
    print("\n🔹 항성 질량 (koi_smass)")
    missing_before = df['koi_smass'].isna().sum()
    if missing_before > 0:
        mass_values, mass_mask = calculate_stellar_mass(df)
        df.loc[mass_values.index, 'koi_smass'] = mass_values
        filled = missing_before - df['koi_smass'].isna().sum()
        print(f"  ✅ {filled}개 채움 (공식: M = R² × 10^(logg - 4.44))")
    
    # 3.2 반장축 (koi_sma)
    print("\n🔹 반장축 (koi_sma)")
    missing_before = df['koi_sma'].isna().sum()
    if missing_before > 0:
        sma_values, sma_mask = calculate_semimajor_axis(df)
        df.loc[sma_values.index, 'koi_sma'] = sma_values
        filled = missing_before - df['koi_sma'].isna().sum()
        print(f"  ✅ {filled}개 채움 (공식: a = (P² × M)^(1/3))")
    
    # 3.3 표면중력 (koi_slogg)
    print("\n🔹 표면중력 (koi_slogg)")
    missing_before = df['koi_slogg'].isna().sum()
    if missing_before > 0:
        logg_values, logg_mask = calculate_surface_gravity(df)
        df.loc[logg_values.index, 'koi_slogg'] = logg_values
        filled = missing_before - df['koi_slogg'].isna().sum()
        print(f"  ✅ {filled}개 채움 (공식: logg = log10(M/R²) + 4.44)")
    
    # 3.4 충격 매개변수 (koi_impact)
    print("\n🔹 충격 매개변수 (koi_impact)")
    missing_before = df['koi_impact'].isna().sum()
    if missing_before > 0:
        impact_values, impact_mask = calculate_impact_parameter(df)
        df.loc[impact_values.index, 'koi_impact'] = impact_values
        filled = missing_before - df['koi_impact'].isna().sum()
        print(f"  ✅ {filled}개 채움 (공식: b = (a/R_star) × cos(i))")
    
    # 3.5 통과 깊이 (koi_depth)
    print("\n🔹 통과 깊이 (koi_depth)")
    missing_before = df['koi_depth'].isna().sum()
    if missing_before > 0:
        depth_values, depth_mask = calculate_transit_depth(df)
        df.loc[depth_values.index, 'koi_depth'] = depth_values
        filled = missing_before - df['koi_depth'].isna().sum()
        print(f"  ✅ {filled}개 채움 (공식: depth = (R_p/R_s)² × 10⁶)")
    
    # 3.6 입사 플럭스 (koi_insol)
    print("\n🔹 입사 플럭스 (koi_insol)")
    missing_before = df['koi_insol'].isna().sum()
    if missing_before > 0:
        insol_values, insol_mask = calculate_insolation_flux(df)
        df.loc[insol_values.index, 'koi_insol'] = insol_values
        filled = missing_before - df['koi_insol'].isna().sum()
        print(f"  ✅ {filled}개 채움 (공식: insol = L_star / a²)")
    
    # 3.7 평형 온도 (koi_teq)
    print("\n🔹 평형 온도 (koi_teq)")
    missing_before = df['koi_teq'].isna().sum()
    if missing_before > 0:
        teq_values, teq_mask = calculate_equilibrium_temperature(df)
        df.loc[teq_values.index, 'koi_teq'] = teq_values
        filled = missing_before - df['koi_teq'].isna().sum()
        print(f"  ✅ {filled}개 채움 (공식: T_eq = T_star × (R_star/2a)^0.5)")
    
    # 4. 회귀 모델로 결측치 채우기
    print("\n🤖 4. 회귀 모델로 결측치 채우기")
    print("-"*100)
    
    # 4.1 항성 반지름 (koi_srad)
    print("\n🔹 항성 반지름 (koi_srad)")
    df = fill_with_regression(
        df, 
        target_col='koi_srad',
        feature_cols=['koi_steff', 'koi_slogg', 'koi_smass'],
        min_samples=100
    )
    
    # 4.2 행성 반지름 (koi_prad)
    print("\n🔹 행성 반지름 (koi_prad)")
    df = fill_with_regression(
        df,
        target_col='koi_prad',
        feature_cols=['koi_depth', 'koi_srad', 'koi_period', 'koi_sma'],
        min_samples=100
    )
    
    # 4.3 유효 온도 (koi_steff)
    print("\n🔹 유효 온도 (koi_steff)")
    df = fill_with_regression(
        df,
        target_col='koi_steff',
        feature_cols=['koi_srad', 'koi_slogg', 'koi_smass'],
        min_samples=100
    )
    
    # 4.4 통과 지속시간 (koi_duration)
    print("\n🔹 통과 지속시간 (koi_duration)")
    df = fill_with_regression(
        df,
        target_col='koi_duration',
        feature_cols=['koi_period', 'koi_srad', 'koi_sma', 'koi_incl'],
        min_samples=100
    )
    
    # 5. 통계적 추정으로 결측치 채우기
    print("\n📊 5. 통계적 추정으로 결측치 채우기")
    print("-"*100)
    
    # 5.1 궤도 이심률 (koi_eccen)
    print("\n🔹 궤도 이심률 (koi_eccen)")
    missing_before = df['koi_eccen'].isna().sum()
    if missing_before > 0:
        # 단주기 행성 (P < 10일): 조석 고정으로 e ≈ 0
        short_period_mask = (df['koi_eccen'].isna()) & (df['koi_period'] < 10)
        df.loc[short_period_mask, 'koi_eccen'] = 0.0
        
        # 장주기 행성: 중앙값으로 채우기
        long_period_mask = (df['koi_eccen'].isna()) & (df['koi_period'] >= 10)
        median_eccen = df.loc[df['koi_period'] >= 10, 'koi_eccen'].median()
        df.loc[long_period_mask, 'koi_eccen'] = median_eccen
        
        filled = missing_before - df['koi_eccen'].isna().sum()
        print(f"  ✅ {filled}개 채움")
        print(f"     - 단주기 (P<10일): 0.0 (조석 고정)")
        print(f"     - 장주기 (P≥10일): {median_eccen:.3f} (중앙값)")
    
    # 5.2 항성 금속성 (koi_smet)
    print("\n🔹 항성 금속성 (koi_smet)")
    missing_before = df['koi_smet'].isna().sum()
    if missing_before > 0:
        # 데이터 출처별 평균값 사용
        for source in ['Kepler', 'K2', 'TESS']:
            mask = (df['koi_smet'].isna()) & (df['data_source'] == source)
            if mask.sum() > 0:
                mean_smet = df.loc[df['data_source'] == source, 'koi_smet'].mean()
                if pd.notna(mean_smet):
                    df.loc[mask, 'koi_smet'] = mean_smet
                else:
                    df.loc[mask, 'koi_smet'] = 0.0  # 태양 금속성
        
        filled = missing_before - df['koi_smet'].isna().sum()
        print(f"  ✅ {filled}개 채움 (데이터 출처별 평균 또는 0.0)")
    
    # 6. 나머지 결측치 처리
    print("\n🔧 6. 나머지 결측치 처리")
    print("-"*100)
    
    # 6.1 근일점 경도 (koi_longp) - 기본값 90°
    if 'koi_longp' in df.columns:
        missing_before = df['koi_longp'].isna().sum()
        if missing_before > 0:
            df['koi_longp'].fillna(90.0, inplace=True)
            print(f"🔹 koi_longp: {missing_before}개 채움 (기본값 90°)")
    
    # 6.2 항성 나이 (koi_sage) - 중앙값
    if 'koi_sage' in df.columns:
        missing_before = df['koi_sage'].isna().sum()
        if missing_before > 0:
            median_sage = df['koi_sage'].median()
            df['koi_sage'].fillna(median_sage, inplace=True)
            print(f"🔹 koi_sage: {missing_before}개 채움 (중앙값 {median_sage:.2f} Gyr)")
    
    # 7. 레이블 결측치 처리 (CANDIDATE로 채우기)
    print("\n🏷️ 7. 레이블 결측치 처리")
    print("-"*100)
    missing_labels = df['koi_disposition'].isna().sum()
    if missing_labels > 0:
        df['koi_disposition'].fillna('CANDIDATE', inplace=True)
        print(f"🔹 koi_disposition: {missing_labels}개를 'CANDIDATE'로 채움")
    
    # 8. 최종 결측치 확인
    print("\n📊 8. 최종 결측치 확인")
    print("-"*100)
    
    final_missing = df.isna().sum()
    final_missing = final_missing[final_missing > 0].sort_values(ascending=False)
    
    if len(final_missing) > 0:
        print("\n⚠️ 남은 결측치:")
        for col, count in final_missing.items():
            pct = count / len(df) * 100
            print(f"  • {col}: {count} ({pct:.2f}%)")
    else:
        print("\n✅ 모든 결측치 처리 완료!")
    
    total_final_missing = df.isna().sum().sum()
    print(f"\n총 결측치: {original_missing:,} → {total_final_missing:,} "
          f"({(1 - total_final_missing/original_missing)*100:.1f}% 감소)")
    
    # 9. 최종 데이터셋 저장
    print("\n💾 9. 최종 데이터셋 저장")
    print("-"*100)
    
    output_path = 'datasets/exoplanets.csv'
    df.to_csv(output_path, index=False)
    
    print(f"✅ 저장 완료: {output_path}")
    print(f"   - 행: {len(df):,}")
    print(f"   - 컬럼: {len(df.columns)}")
    print(f"   - 결측치: {total_final_missing:,}")
    
    # 10. 통계 요약
    print("\n📊 10. 최종 통계")
    print("-"*100)
    
    print("\n🏷️ 레이블 분포:")
    label_counts = df['koi_disposition'].value_counts()
    for label, count in label_counts.items():
        pct = count / len(df) * 100
        print(f"  • {label}: {count:,} ({pct:.1f}%)")
    
    print("\n📍 데이터 출처:")
    source_counts = df['data_source'].value_counts()
    for source, count in source_counts.items():
        pct = count / len(df) * 100
        print(f"  • {source}: {count:,} ({pct:.1f}%)")
    
    print("\n✨ 주요 컬럼 완성도:")
    important_cols = ['koi_period', 'koi_sma', 'koi_prad', 'koi_smass', 
                     'koi_srad', 'koi_steff', 'koi_teq']
    for col in important_cols:
        if col in df.columns:
            completeness = (df[col].notna().sum() / len(df)) * 100
            print(f"  • {col}: {completeness:.1f}%")
    
    print("\n" + "="*100)
    print("✅ 전처리 완료!")
    print("="*100)
    
    return df


if __name__ == "__main__":
    df = preprocess_exoplanet_data()
