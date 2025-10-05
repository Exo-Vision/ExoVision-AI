"""
세 가지 데이터셋(Kepler, K2, TESS)을 통합하고 누락된 값을 계산하여 최종 CSV 파일로 저장
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def calculate_stellar_mass(df, rad_col='st_rad', logg_col='st_logg'):
    """항성 질량 계산"""
    if rad_col not in df.columns or logg_col not in df.columns:
        return None
    logg_sun = 4.44
    mass = (df[rad_col] ** 2) * 10 ** (df[logg_col] - logg_sun)
    return mass.where((mass >= 0.1) & (mass <= 10), np.nan)


def calculate_semimajor_axis(df, period_col='pl_orbper', mass_col='koi_smass'):
    """궤도 반장축 계산"""
    if period_col not in df.columns or mass_col not in df.columns:
        return None
    P_years = df[period_col] / 365.25
    sma = (P_years ** 2 * df[mass_col]) ** (1 / 3)
    return sma.where((sma >= 0.001) & (sma <= 100), np.nan)


def process_kepler(df):
    df = df.copy()
    if 'koi_time0bk' in df:
        df['koi_time0bk'] += 2454833.0
    if {'koi_smass', 'koi_srad', 'koi_slogg'}.issubset(df.columns):
        df['koi_smass'] = df['koi_smass'].fillna(
            calculate_stellar_mass(df, 'koi_srad', 'koi_slogg')
        )
    if {'koi_sma', 'koi_period', 'koi_smass'}.issubset(df.columns):
        df['koi_sma'] = df['koi_sma'].fillna(
            calculate_semimajor_axis(df, 'koi_period', 'koi_smass')
        )
    if 'koi_incl' in df.columns:
        df['koi_incl'] = df['koi_incl'].fillna(89.0)
    df['data_source'] = 'Kepler'
    df['koi_smass_calculated'] = False
    df['koi_sma_calculated'] = False
    df['koi_incl_calculated'] = False
    return df


def process_k2(df):
    df = df.copy()
    if 'disposition' in df:
        df['koi_disposition'] = df['disposition']
    if 'pl_trandep' in df:
        df['koi_depth'] = df['pl_trandep'] * 10000
    if 'pl_trandur' in df:
        df['koi_duration'] = df['pl_trandur'] * 24

    column_mapping = {
        'epic_candname': 'kepoi_name', 'ra': 'ra', 'dec': 'dec',
        'pl_orbper': 'koi_period', 'pl_tranmid': 'koi_time0bk',
        'pl_orbeccen': 'koi_eccen', 'pl_orblper': 'koi_longp',
        'pl_orbincl': 'koi_incl', 'pl_imppar': 'koi_impact',
        'pl_orbsmax': 'koi_sma', 'pl_rade': 'koi_prad', 'pl_insol': 'koi_insol',
        'pl_eqt': 'koi_teq', 'st_rad': 'koi_srad', 'st_mass': 'koi_smass',
        'st_age': 'koi_sage', 'st_teff': 'koi_steff', 'st_logg': 'koi_slogg',
        'st_met': 'koi_smet', 'sy_kepmag': 'koi_kepmag'
    }
    df = df.rename(columns=column_mapping)

    if 'koi_smass' not in df or df['koi_smass'].isna().any():
        if {'koi_srad', 'koi_slogg'}.issubset(df.columns):
            df['koi_smass'] = df['koi_smass'].fillna(
                calculate_stellar_mass(df, 'koi_srad', 'koi_slogg')
            )

    if 'koi_sma' not in df or df['koi_sma'].isna().any():
        if {'koi_period', 'koi_smass'}.issubset(df.columns):
            df['koi_sma'] = df['koi_sma'].fillna(
                calculate_semimajor_axis(df, 'koi_period', 'koi_smass')
            )

    if 'koi_incl' not in df:
        df['koi_incl'] = 89.0
    else:
        df['koi_incl'] = df['koi_incl'].fillna(89.0)

    df['data_source'] = 'K2'
    df['koi_smass_calculated'] = 'st_mass' not in df or df['st_mass'].isna().all()
    df['koi_sma_calculated'] = 'pl_orbsmax' not in df or df['pl_orbsmax'].isna().all()
    df['koi_incl_calculated'] = 'pl_orbincl' not in df or df['pl_orbincl'].isna().all()
    return df


def process_tess(df):
    df = df.copy()
    if 'tfopwg_disp' in df:
        label_mapping = {'CP': 'CONFIRMED', 'KP': 'CONFIRMED', 'FP': 'FALSE POSITIVE', 'PC': 'CANDIDATE'}
        df['koi_disposition'] = df['tfopwg_disp'].map(label_mapping)

    column_mapping = {
        'toi': 'kepoi_name', 'ra': 'ra', 'dec': 'dec',
        'pl_orbper': 'koi_period', 'pl_tranmid': 'koi_time0bk',
        'pl_trandurh': 'koi_duration', 'pl_trandep': 'koi_depth',
        'pl_rade': 'koi_prad', 'pl_insol': 'koi_insol', 'pl_eqt': 'koi_teq',
        'st_rad': 'koi_srad', 'st_teff': 'koi_steff', 'st_logg': 'koi_slogg'
    }
    df = df.rename(columns=column_mapping)

    df['koi_smass'] = calculate_stellar_mass(df, 'koi_srad', 'koi_slogg')
    df['koi_sma'] = calculate_semimajor_axis(df, 'koi_period', 'koi_smass')
    df['koi_incl'] = 89.0

    df['koi_eccen'] = 0.0
    df['koi_longp'] = 90.0
    df['koi_smet'] = 0.0
    df['koi_sage'] = 5.0
    df['koi_kepmag'] = np.nan
    df['koi_ingress'] = df['koi_duration'] * 0.1

    df['data_source'] = 'TESS'
    df['koi_smass_calculated'] = True
    df['koi_sma_calculated'] = True
    df['koi_incl_calculated'] = True
    return df


def integrate_datasets():
    df_kepler = pd.read_csv('datasets/kepler_merged.csv')
    df_k2 = pd.read_csv('datasets/k2_merged.csv')
    df_tess = pd.read_csv('datasets/tess_merged.csv')

    df_kepler = process_kepler(df_kepler)
    df_k2 = process_k2(df_k2)
    df_tess = process_tess(df_tess)

    essential_columns = [
        'kepoi_name', 'koi_disposition', 'ra', 'dec', 'koi_period', 'koi_time0bk',
        'koi_eccen', 'koi_longp', 'koi_incl', 'koi_impact', 'koi_sma',
        'koi_duration', 'koi_depth', 'koi_ingress', 'koi_prad', 'koi_insol',
        'koi_teq', 'koi_srad', 'koi_smass', 'koi_sage', 'koi_steff', 'koi_slogg',
        'koi_smet', 'koi_kepmag', 'data_source',
        'koi_smass_calculated', 'koi_sma_calculated', 'koi_incl_calculated'
    ]

    def select(df): return df.reindex(columns=essential_columns)

    df_final = pd.concat([select(df_kepler), select(df_k2), select(df_tess)], ignore_index=True)
    df_final.to_csv('datasets/exoplanets_integrated.csv', index=False)
    return df_final


if __name__ == "__main__":
    integrate_datasets()