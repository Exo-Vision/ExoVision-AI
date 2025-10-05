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


def validate_expert_input(expert_input: Dict) -> pd.DataFrame:
    """
    전문가 입력 검증 및 DataFrame 변환
    
    Args:
        expert_input: 29개 파라미터 딕셔너리
        
    Returns:
        29개 컬럼을 가진 DataFrame (모델 학습 시 순서와 동일)
    """
    columns_order = [
        "ra", "dec", "koi_period", "koi_eccen", "koi_longp", "koi_incl",
        "koi_impact", "koi_sma", "koi_duration", "koi_depth", "koi_prad",
        "koi_insol", "koi_teq", "koi_srad", "koi_smass", "koi_sage",
        "koi_steff", "koi_slogg", "koi_smet", "planet_star_ratio",
        "orbital_energy", "transit_signal", "stellar_density",
        "planet_density_proxy", "log_period", "log_depth", "log_insol",
        "orbit_stability", "transit_snr"
    ]
    
    df = pd.DataFrame([{col: expert_input[col] for col in columns_order}])
    
    return df
