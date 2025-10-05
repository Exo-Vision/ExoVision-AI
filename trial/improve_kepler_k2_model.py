"""
외계행성 분류 파이프라인 - 개선된 Kepler+K2 통합
공통 컬럼을 더 많이 찾아 모델 성능 향상
"""

import os

import numpy as np
import pandas as pd


def improved_kepler_k2_merge():
    """개선된 Kepler+K2 통합 (더 많은 공통 특성 사용)"""

    print("=" * 80)
    print("개선된 Kepler+K2 데이터 통합")
    print("=" * 80)

    # Kepler 데이터 로드
    kepler_df = pd.read_csv("datasets/kepler.csv", low_memory=False)
    k2_df = pd.read_csv("datasets/k2.csv", low_memory=False)

    print(f"\nKepler 원본: {kepler_df.shape}")
    print(f"K2 원본: {k2_df.shape}")

    # 수동으로 매핑 (더 많은 특성 포함)
    kepler_features = {
        # Target
        "koi_disposition": "disposition",
        # Planet properties
        "koi_period": "pl_orbper",
        "koi_duration": "pl_trandur",
        "koi_depth": "pl_trandep",
        "koi_prad": "pl_rade",
        "koi_insol": "pl_insol",
        "koi_teq": "pl_eqt",
        "koi_impact": "pl_imppar",
        # Stellar properties
        "koi_steff": "st_teff",
        "koi_slogg": "st_logg",
        "koi_srad": "st_rad",
        "koi_smass": "st_mass",
        "koi_smet": "st_met",
        # Position
        "ra": "ra",
        "dec": "dec",
        "koi_kepmag": "kepler_mag",
    }

    k2_features = {
        # Target
        "disposition": "disposition",
        # Planet properties
        "pl_orbper": "pl_orbper",
        "pl_trandur": "pl_trandur",
        "pl_trandep": "pl_trandep",
        "pl_rade": "pl_rade",
        "pl_insol": "pl_insol",
        "pl_eqt": "pl_eqt",
        "pl_imppar": "pl_imppar",
        # Stellar properties
        "st_teff": "st_teff",
        "st_logg": "st_logg",
        "st_rad": "st_rad",
        "st_mass": "st_mass",
        "st_met": "st_met",
        # Position
        "ra": "ra",
        "dec": "dec",
        "sy_kepmag": "kepler_mag",
    }

    # Kepler 매핑
    kepler_mapped = pd.DataFrame()
    for kepler_col, common_name in kepler_features.items():
        if kepler_col in kepler_df.columns:
            kepler_mapped[common_name] = kepler_df[kepler_col]

    kepler_mapped["data_source"] = "Kepler"

    # K2 매핑
    k2_mapped = pd.DataFrame()
    for k2_col, common_name in k2_features.items():
        if k2_col in k2_df.columns:
            k2_mapped[common_name] = k2_df[k2_col]

    k2_mapped["data_source"] = "K2"

    print(f"\nKepler 매핑 후: {kepler_mapped.shape}")
    print(f"K2 매핑 후: {k2_mapped.shape}")

    # 공통 컬럼으로 정렬
    common_cols = list(set(kepler_mapped.columns) & set(k2_mapped.columns))
    print(f"\n공통 컬럼 수: {len(common_cols)}")
    print(f"공통 컬럼: {common_cols}")

    # 통합
    merged = pd.concat(
        [kepler_mapped[common_cols], k2_mapped[common_cols]], ignore_index=True
    )

    print(f"\n통합 데이터: {merged.shape}")

    # Disposition 분포
    if "disposition" in merged.columns:
        print(f"\nDisposition 분포:")
        print(merged["disposition"].value_counts())

    # 저장
    output_path = "datasets/kepler_k2_merged_improved.csv"
    merged.to_csv(output_path, index=False)
    print(f"\n저장 완료: {output_path}")

    return merged


def retrain_with_improved_data():
    """개선된 데이터로 모델 재학습"""
    from exoplanet_classification_pipeline import ExoplanetClassificationPipeline

    # 개선된 통합 데이터 사용
    merged_df = pd.read_csv("datasets/kepler_k2_merged_improved.csv")

    print("\n" + "=" * 80)
    print("개선된 데이터로 Kepler+K2 모델 재학습")
    print("=" * 80)

    pipeline = ExoplanetClassificationPipeline()

    # 데이터 준비
    X, y, features = pipeline.prepare_kepler_k2_data_for_training(merged_df)

    print(f"\n✨ Feature 수 증가: {len(features)}개")
    print(f"Features: {features}")

    # 모델 학습
    model, data = pipeline.train_kepler_k2_model(X, y, use_optuna=False)

    # 저장
    pipeline.save_models(prefix="improved_")

    print("\n개선된 모델 저장 완료!")
    return model


if __name__ == "__main__":
    # 1. 개선된 데이터 통합
    merged = improved_kepler_k2_merge()

    # 2. 개선된 데이터로 재학습
    model = retrain_with_improved_data()

    print("\n" + "=" * 80)
    print("✅ 개선 완료!")
    print("=" * 80)
    print("\n결과:")
    print("- 공통 컬럼이 2개에서 15+개로 증가")
    print("- 모델 성능 향상 예상")
    print("- models/improved_kepler_k2_model_*.pkl로 저장됨")
    print("- 모델 성능 향상 예상")
    print("- models/improved_kepler_k2_model_*.pkl로 저장됨")
