"""
Feature engineering utilities for beginner mode
초보자 모드에서 5개 파라미터로부터 나머지 피처 생성
"""
import pandas as pd
from typing import Dict
from app.core.config import settings


def create_full_features_from_beginner(beginner_input: Dict) -> pd.DataFrame:
    """
    초보자 입력 5개 파라미터로부터 전체 28개 피처 생성
    
    Args:
        beginner_input: {
            "koi_prad": float,
            "dec": float,
            "koi_smet": float,
            "planet_star_ratio": float,
            "planet_density_proxy": float
        }
    
    Returns:
        28개 컬럼을 가진 DataFrame
    """
    # 초보자가 제공한 값
    user_values = {
        "koi_prad": beginner_input["koi_prad"],
        "dec": beginner_input["dec"],
        "koi_smet": beginner_input["koi_smet"],
        "planet_star_ratio": beginner_input["planet_star_ratio"],
        "planet_density_proxy": beginner_input["planet_density_proxy"],
    }
    
    # 기본값으로 채우기
    full_data = {**settings.BEGINNER_DEFAULTS, **user_values}
    
    # DataFrame 생성 (컬럼 순서 보장 - 모델 학습 시 순서와 동일)
    columns_order = [
        "ra", "dec", "koi_period", "koi_eccen", "koi_longp", "koi_incl",
        "koi_impact", "koi_sma", "koi_duration", "koi_depth", "koi_prad",
        "koi_insol", "koi_teq", "koi_srad", "koi_smass", "koi_sage",
        "koi_steff", "koi_slogg", "koi_smet", "planet_star_ratio",
        "orbital_energy", "transit_signal", "stellar_density",
        "planet_density_proxy", "log_period", "log_depth", "log_insol",
        "orbit_stability", "transit_snr"
    ]
    
    df = pd.DataFrame([{col: full_data[col] for col in columns_order}])
    
    return df


def create_full_features_from_expert(expert_input: Dict) -> pd.DataFrame:
    """
    전문가 입력 19개 파라미터로부터 전체 29개 피처 생성
    나머지 10개 엔지니어링 피처는 자동 계산

    Args:
        expert_input: 19개 기본 파라미터 딕셔너리

    Returns:
        29개 컬럼을 가진 DataFrame (모델 학습 시 순서와 동일)
    """
    import numpy as np

    # 엔지니어링 피처 계산
    planet_star_ratio = expert_input["koi_prad"] / expert_input["koi_srad"] if expert_input["koi_srad"] > 0 else 0.015

    # orbital_energy = period / semi-major axis (근사)
    orbital_energy = expert_input["koi_period"] / expert_input["koi_sma"] if expert_input["koi_sma"] > 0 else 1.0

    # transit_signal = depth * duration (통과 신호 강도)
    transit_signal = expert_input["koi_depth"] * expert_input["koi_duration"]

    # stellar_density = mass / radius^3 (별 밀도 근사)
    stellar_density = expert_input["koi_smass"] / (expert_input["koi_srad"] ** 3) if expert_input["koi_srad"] > 0 else 1.0

    # planet_density_proxy = planet_radius / semi-major axis
    planet_density_proxy = expert_input["koi_prad"] / expert_input["koi_sma"] if expert_input["koi_sma"] > 0 else 0.5

    # 로그 변환 (0이 되지 않도록 처리)
    log_period = np.log10(max(expert_input["koi_period"], 0.1))
    log_depth = np.log10(max(expert_input["koi_depth"], 0.1))
    log_insol = np.log10(max(expert_input["koi_insol"], 0.1))

    # orbit_stability = eccentricity * impact parameter (궤도 안정성 근사)
    orbit_stability = expert_input["koi_eccen"] * expert_input["koi_impact"]

    # transit_snr = depth / sqrt(duration) (신호 대 잡음비 근사)
    transit_snr = expert_input["koi_depth"] / np.sqrt(max(expert_input["koi_duration"], 0.1))

    # 전체 피처 구성
    full_data = {
        **expert_input,
        "planet_star_ratio": planet_star_ratio,
        "orbital_energy": orbital_energy,
        "transit_signal": transit_signal,
        "stellar_density": stellar_density,
        "planet_density_proxy": planet_density_proxy,
        "log_period": log_period,
        "log_depth": log_depth,
        "log_insol": log_insol,
        "orbit_stability": orbit_stability,
        "transit_snr": transit_snr
    }

    # DataFrame 생성 (컬럼 순서 보장)
    columns_order = [
        "ra", "dec", "koi_period", "koi_eccen", "koi_longp", "koi_incl",
        "koi_impact", "koi_sma", "koi_duration", "koi_depth", "koi_prad",
        "koi_insol", "koi_teq", "koi_srad", "koi_smass", "koi_sage",
        "koi_steff", "koi_slogg", "koi_smet", "planet_star_ratio",
        "orbital_energy", "transit_signal", "stellar_density",
        "planet_density_proxy", "log_period", "log_depth", "log_insol",
        "orbit_stability", "transit_snr"
    ]

    df = pd.DataFrame([{col: full_data[col] for col in columns_order}])

    return df
