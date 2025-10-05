"""
세 데이터셋의 에러 컬럼을 권장 전략에 따라 통합하는 스크립트
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 100)
print("외계행성 데이터셋 에러 컬럼 통합 시작")
print("=" * 100)
print()


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
        'keep_lim': lim 고려 (K2/TESS만)
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
    
    elif strategy == 'keep_lim':
        # lim 플래그 보존
        if lim_col and lim_col in df.columns:
            df[base_col + '_limit_flag'] = df[lim_col]
        # 평균 에러도 계산
        df[base_col + '_error'] = (
            df[err1_col].abs() + df[err2_col].abs()
        ) / 2
    
    return df


def process_kepler(input_file, output_file):
    """Kepler 데이터셋 처리"""
    print("📊 Kepler 데이터셋 처리 중...")
    
    df = pd.read_csv(input_file)
    print(f"   - 원본 데이터: {len(df)} 행, {len(df.columns)} 컬럼")
    
    # 컬럼별 권장 전략 적용
    columns_to_merge = {
        # 궤도 파라미터
        'koi_period': 'average',      # 궤도 주기: 평균
        'koi_time0bk': 'average',     # 통과 시각 (BKJD): 평균
        'koi_time0': 'average',       # 통과 시각 (BJD): 평균
        'koi_eccen': 'separate',      # 이심률: 분리 보존
        'koi_longp': 'average',       # 근점 경도: 평균
        'koi_impact': 'average',      # 충돌 파라미터: 평균
        'koi_duration': 'average',    # 통과 지속시간: 평균
        'koi_ingress': 'average',     # 진입 시간: 평균
        'koi_depth': 'average',       # 통과 깊이: 평균
        'koi_ror': 'average',         # 반지름 비율: 평균
        'koi_srho': 'max',            # 항성 밀도: 최대
        
        # 행성 파라미터
        'koi_prad': 'average',        # 행성 반지름: 평균
        'koi_sma': 'average',         # 장반경: 평균
        'koi_incl': 'average',        # 경사각: 평균
        'koi_teq': 'average',         # 평형 온도: 평균
        'koi_insol': 'average',       # 복사 플럭스: 평균
        'koi_dor': 'average',         # 거리/반지름 비율: 평균
        
        # 항성 파라미터
        'koi_steff': 'max',           # 항성 온도: 최대
        'koi_slogg': 'max',           # 항성 중력: 최대
        'koi_smet': 'average',        # 항성 금속 함량: 평균
        'koi_srad': 'average',        # 항성 반지름: 평균
        'koi_smass': 'max',           # 항성 질량: 최대
        'koi_sage': 'average',        # 항성 나이: 평균
    }
    
    for col, strategy in columns_to_merge.items():
        df = merge_error_columns(df, col, strategy=strategy, dataset='kepler')
    
    # 원본 에러 컬럼 제거
    error_cols_to_drop = [c for c in df.columns if c.endswith('_err1') or c.endswith('_err2')]
    df = df.drop(columns=error_cols_to_drop)
    
    df.to_csv(output_file, index=False)
    print(f"   ✅ 완료: {len(df)} 행, {len(df.columns)} 컬럼")
    print(f"   - 저장 위치: {output_file}")
    print()
    
    return df


def process_k2(input_file, output_file):
    """K2 데이터셋 처리"""
    print("📊 K2 데이터셋 처리 중...")
    
    df = pd.read_csv(input_file)
    print(f"   - 원본 데이터: {len(df)} 행, {len(df.columns)} 컬럼")
    
    # 컬럼별 권장 전략 적용
    columns_to_merge = {
        # 궤도 파라미터
        'pl_orbper': 'average',       # 궤도 주기: 평균
        'pl_orbsmax': 'average',      # 장반경: 평균
        'pl_orbeccen': 'separate',    # 이심률: 분리 보존
        'pl_orbincl': 'average',      # 경사각: 평균
        'pl_tranmid': 'average',      # 통과 중심 시각: 평균
        'pl_imppar': 'average',       # 충돌 파라미터: 평균
        
        # 행성 물리량
        'pl_rade': 'average',         # 행성 반지름 (지구): 평균
        'pl_radj': 'average',         # 행성 반지름 (목성): 평균
        'pl_masse': 'keep_lim',       # 행성 질량 (지구): lim 고려
        'pl_massj': 'keep_lim',       # 행성 질량 (목성): lim 고려
        'pl_msinie': 'keep_lim',      # 질량*sin(i) (지구): lim 고려
        'pl_msinij': 'keep_lim',      # 질량*sin(i) (목성): lim 고려
        'pl_cmasse': 'keep_lim',      # 계산 질량 (지구): lim 고려
        'pl_cmassj': 'keep_lim',      # 계산 질량 (목성): lim 고려
        'pl_bmasse': 'keep_lim',      # 최적 질량 (지구): lim 고려
        'pl_bmassj': 'keep_lim',      # 최적 질량 (목성): lim 고려
        'pl_dens': 'average',         # 행성 밀도: 평균
        'pl_insol': 'average',        # 복사 플럭스: 평균
        'pl_eqt': 'average',          # 평형 온도: 평균
        
        # 통과 파라미터
        'pl_trandep': 'average',      # 통과 깊이: 평균
        'pl_trandur': 'average',      # 통과 지속시간: 평균
        'pl_ratdor': 'average',       # 거리/반지름 비율: 평균
        'pl_ratror': 'average',       # 반지름 비율: 평균
        'pl_occdep': 'average',       # 가림 깊이: 평균
        
        # 기타 궤도 파라미터
        'pl_orbtper': 'average',      # 근점 시각: 평균
        'pl_orblper': 'average',      # 근점 인수: 평균
        'pl_rvamp': 'average',        # 시선속도 진폭: 평균
        'pl_projobliq': 'average',    # 투영 경사각: 평균
        'pl_trueobliq': 'average',    # 실제 경사각: 평균
        
        # 항성 파라미터
        'st_teff': 'max',             # 항성 온도: 최대
        'st_rad': 'average',          # 항성 반지름: 평균
        'st_mass': 'max',             # 항성 질량: 최대
        'st_met': 'average',          # 항성 금속 함량: 평균
        'st_lum': 'average',          # 항성 광도: 평균
        'st_logg': 'max',             # 항성 중력: 최대
        'st_age': 'average',          # 항성 나이: 평균
        'st_dens': 'average',         # 항성 밀도: 평균
        'st_vsin': 'average',         # 항성 회전속도: 평균
        'st_rotp': 'average',         # 항성 회전주기: 평균
        'st_radv': 'average',         # 시선속도: 평균
        
        # 시스템 파라미터
        'sy_pm': 'average',           # 고유운동: 평균
        'sy_pmra': 'average',         # 고유운동 (RA): 평균
        'sy_pmdec': 'average',        # 고유운동 (Dec): 평균
        'sy_dist': 'average',         # 거리: 평균
        'sy_plx': 'average',          # 시차: 평균
        
        # 측광 등급
        'sy_bmag': 'average',
        'sy_vmag': 'average',
        'sy_jmag': 'average',
        'sy_hmag': 'average',
        'sy_kmag': 'average',
        'sy_umag': 'average',
        'sy_gmag': 'average',
        'sy_rmag': 'average',
        'sy_imag': 'average',
        'sy_zmag': 'average',
        'sy_w1mag': 'average',
        'sy_w2mag': 'average',
        'sy_w3mag': 'average',
        'sy_w4mag': 'average',
        'sy_gaiamag': 'average',
        'sy_icmag': 'average',
        'sy_tmag': 'average',
        'sy_kepmag': 'average',
    }
    
    for col, strategy in columns_to_merge.items():
        df = merge_error_columns(df, col, strategy=strategy, dataset='k2')
    
    # 원본 에러 컬럼 제거 (lim은 keep_lim 전략에서 이미 처리됨)
    error_cols_to_drop = [c for c in df.columns if c.endswith('err1') or c.endswith('err2')]
    # lim 컬럼도 제거 (이미 _limit_flag로 변환됨)
    lim_cols_to_drop = [c for c in df.columns if c.endswith('lim') and not c.endswith('_limit_flag')]
    df = df.drop(columns=error_cols_to_drop + lim_cols_to_drop, errors='ignore')
    
    df.to_csv(output_file, index=False)
    print(f"   ✅ 완료: {len(df)} 행, {len(df.columns)} 컬럼")
    print(f"   - 저장 위치: {output_file}")
    print()
    
    return df


def process_tess(input_file, output_file):
    """TESS 데이터셋 처리"""
    print("📊 TESS 데이터셋 처리 중...")
    
    df = pd.read_csv(input_file)
    print(f"   - 원본 데이터: {len(df)} 행, {len(df.columns)} 컬럼")
    
    # 컬럼별 권장 전략 적용
    columns_to_merge = {
        # 위치
        'ra': 'average',
        'dec': 'average',
        
        # 고유운동
        'st_pmra': 'average',
        'st_pmdec': 'average',
        
        # 행성 파라미터
        'pl_tranmid': 'average',      # 통과 중심 시각: 평균
        'pl_orbper': 'average',       # 궤도 주기: 평균
        'pl_trandurh': 'average',     # 통과 지속시간: 평균
        'pl_trandep': 'average',      # 통과 깊이: 평균
        'pl_rade': 'average',         # 행성 반지름: 평균
        'pl_insol': 'average',        # 복사 플럭스: 평균
        'pl_eqt': 'average',          # 평형 온도: 평균
        
        # 항성 파라미터
        'st_tmag': 'average',         # TESS 등급: 평균
        'st_dist': 'average',         # 거리: 평균
        'st_teff': 'max',             # 항성 온도: 최대
        'st_logg': 'max',             # 항성 중력: 최대
        'st_rad': 'average',          # 항성 반지름: 평균
    }
    
    for col, strategy in columns_to_merge.items():
        df = merge_error_columns(df, col, strategy=strategy, dataset='tess')
    
    # 원본 에러 컬럼 제거
    error_cols_to_drop = [c for c in df.columns if c.endswith('err1') or c.endswith('err2')]
    lim_cols_to_drop = [c for c in df.columns if c.endswith('lim') and not c.endswith('_limit_flag')]
    df = df.drop(columns=error_cols_to_drop + lim_cols_to_drop, errors='ignore')
    
    df.to_csv(output_file, index=False)
    print(f"   ✅ 완료: {len(df)} 행, {len(df.columns)} 컬럼")
    print(f"   - 저장 위치: {output_file}")
    print()
    
    return df


if __name__ == "__main__":
    # 데이터셋 처리
    datasets_dir = Path('datasets')
    output_dir = Path('datasets')
    
    # Kepler 처리
    df_kepler = process_kepler(
        datasets_dir / 'kepler.csv',
        output_dir / 'kepler_merged.csv'
    )
    
    # K2 처리
    df_k2 = process_k2(
        datasets_dir / 'k2.csv',
        output_dir / 'k2_merged.csv'
    )
    
    # TESS 처리
    df_tess = process_tess(
        datasets_dir / 'tess.csv',
        output_dir / 'tess_merged.csv'
    )
    
    print("=" * 100)
    print("✅ 모든 데이터셋 통합 완료!")
    print("=" * 100)
    print()
    print("생성된 파일:")
    print("  - datasets/kepler_merged.csv")
    print("  - datasets/k2_merged.csv")
    print("  - datasets/tess_merged.csv")
    print()
    print("다음 단계:")
    print("  1. 각 데이터셋의 컬럼 설명 문서 확인")
    print("  2. 데이터셋 통합 (kepler_merged + k2_merged + tess_merged)")
    print("  3. 머신러닝 모델 학습")
