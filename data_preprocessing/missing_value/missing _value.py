"""
세 데이터셋의 에러 컬럼을 권장 전략에 따라 통합하는 스크립트
"""

import pandas as pd
import numpy as np
from pathlib import Path


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
    
    # 에러 컬럼이 존재하지 않으면 그대로 반환
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
        # 평균 에러 계산
        df[base_col + '_error'] = (
            df[err1_col].abs() + df[err2_col].abs()
        ) / 2
    
    return df


def process_kepler(input_file, output_file):
    """Kepler 데이터셋의 에러 컬럼 통합"""
    df = pd.read_csv(input_file)
    
    # 컬럼별 권장 전략 적용
    columns_to_merge = {
        # 궤도 파라미터
        'koi_period': 'average',
        'koi_time0bk': 'average',
        'koi_time0': 'average',
        'koi_eccen': 'separate',
        'koi_longp': 'average',
        'koi_impact': 'average',
        'koi_duration': 'average',
        'koi_ingress': 'average',
        'koi_depth': 'average',
        'koi_ror': 'average',
        'koi_srho': 'max',
        # 행성 파라미터
        'koi_prad': 'average',
        'koi_sma': 'average',
        'koi_incl': 'average',
        'koi_teq': 'average',
        'koi_insol': 'average',
        'koi_dor': 'average',
        # 항성 파라미터
        'koi_steff': 'max',
        'koi_slogg': 'max',
        'koi_smet': 'average',
        'koi_srad': 'average',
        'koi_smass': 'max',
        'koi_sage': 'average',
    }
    
    for col, strategy in columns_to_merge.items():
        df = merge_error_columns(df, col, strategy=strategy, dataset='kepler')
    
    # 원본 에러 컬럼 제거
    error_cols_to_drop = [c for c in df.columns if c.endswith('_err1') or c.endswith('_err2')]
    df = df.drop(columns=error_cols_to_drop)
    
    # 결과 저장
    df.to_csv(Path("datasets") / "kepler_merged.csv", index=False)
    return df


def process_k2(input_file, output_file):
    """K2 데이터셋의 에러 컬럼 통합"""
    df = pd.read_csv(input_file)
    
    # 컬럼별 권장 전략 적용
    columns_to_merge = {
        # 궤도 파라미터
        'pl_orbper': 'average',
        'pl_orbsmax': 'average',
        'pl_orbeccen': 'separate',
        'pl_orbincl': 'average',
        'pl_tranmid': 'average',
        'pl_imppar': 'average',
        # 행성 물리량
        'pl_rade': 'average',
        'pl_radj': 'average',
        'pl_masse': 'keep_lim',
        'pl_massj': 'keep_lim',
        'pl_msinie': 'keep_lim',
        'pl_msinij': 'keep_lim',
        'pl_cmasse': 'keep_lim',
        'pl_cmassj': 'keep_lim',
        'pl_bmasse': 'keep_lim',
        'pl_bmassj': 'keep_lim',
        'pl_dens': 'average',
        'pl_insol': 'average',
        'pl_eqt': 'average',
        # 통과 파라미터
        'pl_trandep': 'average',
        'pl_trandur': 'average',
        'pl_ratdor': 'average',
        'pl_ratror': 'average',
        'pl_occdep': 'average',
        # 기타 궤도 파라미터
        'pl_orbtper': 'average',
        'pl_orblper': 'average',
        'pl_rvamp': 'average',
        'pl_projobliq': 'average',
        'pl_trueobliq': 'average',
        # 항성 파라미터
        'st_teff': 'max',
        'st_rad': 'average',
        'st_mass': 'max',
        'st_met': 'average',
        'st_lum': 'average',
        'st_logg': 'max',
        'st_age': 'average',
        'st_dens': 'average',
        'st_vsin': 'average',
        'st_rotp': 'average',
        'st_radv': 'average',
        # 시스템 파라미터
        'sy_pm': 'average',
        'sy_pmra': 'average',
        'sy_pmdec': 'average',
        'sy_dist': 'average',
        'sy_plx': 'average',
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
    
    # 원본 에러 및 lim 컬럼 제거
    error_cols_to_drop = [c for c in df.columns if c.endswith('err1') or c.endswith('err2')]
    lim_cols_to_drop = [c for c in df.columns if c.endswith('lim') and not c.endswith('_limit_flag')]
    df = df.drop(columns=error_cols_to_drop + lim_cols_to_drop, errors='ignore')
    
    # 결과 저장
    df.to_csv(Path("datasets") / "k2_merged.csv", index=False)
    return df


def process_tess(input_file, output_file):
    """TESS 데이터셋의 에러 컬럼 통합"""
    df = pd.read_csv(input_file)
    
    # 컬럼별 권장 전략 적용
    columns_to_merge = {
        # 위치
        'ra': 'average',
        'dec': 'average',
        # 고유운동
        'st_pmra': 'average',
        'st_pmdec': 'average',
        # 행성 파라미터
        'pl_tranmid': 'average',
        'pl_orbper': 'average',
        'pl_trandurh': 'average',
        'pl_trandep': 'average',
        'pl_rade': 'average',
        'pl_insol': 'average',
        'pl_eqt': 'average',
        # 항성 파라미터
        'st_tmag': 'average',
        'st_dist': 'average',
        'st_teff': 'max',
        'st_logg': 'max',
        'st_rad': 'average',
    }
    
    for col, strategy in columns_to_merge.items():
        df = merge_error_columns(df, col, strategy=strategy, dataset='tess')
    
    # 원본 에러 및 lim 컬럼 제거
    error_cols_to_drop = [c for c in df.columns if c.endswith('err1') or c.endswith('err2')]
    lim_cols_to_drop = [c for c in df.columns if c.endswith('lim') and not c.endswith('_limit_flag')]
    df = df.drop(columns=error_cols_to_drop + lim_cols_to_drop, errors='ignore')
    
    # 결과 저장
    df.to_csv(Path("datasets") / "tess_merged.csv", index=False)
    return df


if __name__ == "__main__":
    datasets_dir = Path('datasets')

    # Kepler 처리
    df_kepler = process_kepler(
        datasets_dir / 'kepler.csv',
        Path("datasets") / 'kepler_merged.csv'
    )

    # K2 처리
    df_k2 = process_k2(
        datasets_dir / 'k2.csv',
        Path("datasets") / 'k2_merged.csv'
    )

    # TESS 처리
    df_tess = process_tess(
        datasets_dir / 'tess.csv',
        Path("datasets") / 'tess_merged.csv'
    )
