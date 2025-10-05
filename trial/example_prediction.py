"""
외계행성 분류 파이프라인 - 모델 로드 및 예측 예제
저장된 모델을 로드하여 새로운 데이터에 대해 예측하는 방법
"""

import numpy as np
import pandas as pd

from exoplanet_classification_pipeline import ExoplanetClassificationPipeline


def predict_with_saved_models():
    """저장된 모델로 예측하기"""

    # 파이프라인 인스턴스 생성
    pipeline = ExoplanetClassificationPipeline(data_dir="datasets", models_dir="models")

    # 저장된 모델 로드 (파일명을 실제 생성된 파일명으로 변경하세요)
    print("모델 로드 중...")

    # 예시: models 폴더에서 가장 최신 모델 찾기
    import glob
    import os

    tess_models = glob.glob(os.path.join("models", "tess_model_*.pkl"))
    kepler_models = glob.glob(os.path.join("models", "kepler_k2_model_*.pkl"))

    if tess_models and kepler_models:
        # 가장 최신 파일 선택
        latest_tess = max(tess_models, key=os.path.getctime)
        latest_kepler = max(kepler_models, key=os.path.getctime)

        print(f"TESS 모델 로드: {latest_tess}")
        print(f"Kepler+K2 모델 로드: {latest_kepler}")

        pipeline.load_models(
            tess_model_path=latest_tess, kepler_k2_model_path=latest_kepler
        )

        # 예측 예제
        print("\n" + "=" * 80)
        print("예측 예제")
        print("=" * 80)

        # TESS 데이터로 예측 (외계행성 후보 여부)
        print("\n1. TESS 모델로 외계행성 후보 예측")
        tess_data = pd.read_csv("datasets/tess_augmented.csv")

        # Feature 준비 (실제로는 prepare_tess_data_for_training 함수 사용)
        feature_cols = [
            col
            for col in tess_data.columns
            if col not in ["toi", "tid", "tfopwg_disp", "is_candidate"]
            and tess_data[col].dtype in ["float64", "int64"]
        ]

        X_sample = tess_data[feature_cols].head(10).fillna(0)
        X_sample = X_sample.replace([np.inf, -np.inf], 0)

        predictions = pipeline.tess_model.predict(X_sample)
        probabilities = pipeline.tess_model.predict_proba(X_sample)

        print("\n샘플 예측 결과 (TESS):")
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            print(
                f"샘플 {i+1}: {'후보' if pred == 1 else '비후보'} "
                f"(확률: {prob[1]:.2%})"
            )

        # Kepler+K2 데이터로 예측 (후보 → 확정 여부)
        print("\n2. Kepler+K2 모델로 확정 외계행성 예측")
        kepler_data = pd.read_csv("datasets/kepler_k2_merged.csv")

        # CANDIDATE만 필터링
        candidates = kepler_data[kepler_data["koi_disposition"] == "CANDIDATE"]

        if len(candidates) > 0:
            feature_cols_kepler = [
                col
                for col in candidates.columns
                if col
                not in [
                    "kepid",
                    "kepoi_name",
                    "kepler_name",
                    "koi_disposition",
                    "koi_pdisposition",
                ]
                and candidates[col].dtype in ["float64", "int64"]
            ]

            X_candidates = candidates[feature_cols_kepler].head(10).fillna(0)
            X_candidates = X_candidates.replace([np.inf, -np.inf], 0)

            predictions = pipeline.kepler_k2_model.predict(X_candidates)
            probabilities = pipeline.kepler_k2_model.predict_proba(X_candidates)

            print("\n샘플 예측 결과 (Kepler+K2):")
            for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                print(
                    f"후보 {i+1}: {'확정 가능성 높음' if pred == 1 else '후보 단계'} "
                    f"(확률: {prob[1]:.2%})"
                )

        print("\n" + "=" * 80)
        print("예측 완료!")
        print("=" * 80)

    else:
        print("저장된 모델을 찾을 수 없습니다.")
        print("먼저 파이프라인을 실행하여 모델을 학습하세요:")
        print("  python exoplanet_classification_pipeline.py")


def create_prediction_function():
    """새로운 데이터를 예측하는 함수 예제"""

    def predict_exoplanet_candidate(features_dict):
        """
        단일 관측 데이터로 외계행성 후보 예측

        Parameters:
        -----------
        features_dict : dict
            관측 특성 딕셔너리
            예: {
                'pl_orbper': 10.5,
                'pl_rade': 1.2,
                'st_teff': 5800,
                ...
            }

        Returns:
        --------
        prediction : int
            0 (비후보) 또는 1 (후보)
        probability : float
            후보일 확률
        """
        pipeline = ExoplanetClassificationPipeline()

        # 모델 로드 (최신 모델 자동 선택)
        import glob
        import os

        tess_models = glob.glob("models/tess_model_*.pkl")
        if tess_models:
            latest_model = max(tess_models, key=os.path.getctime)
            pipeline.load_models(tess_model_path=latest_model)

            # 예측
            X = pd.DataFrame([features_dict])
            X = X.fillna(0).replace([np.inf, -np.inf], 0)

            prediction = pipeline.tess_model.predict(X)[0]
            probability = pipeline.tess_model.predict_proba(X)[0][1]

            return prediction, probability
        else:
            raise FileNotFoundError("저장된 모델이 없습니다.")

    # 사용 예제
    print("\n" + "=" * 80)
    print("단일 예측 함수 사용 예제")
    print("=" * 80)

    # 예제 데이터
    example_observation = {
        "pl_orbper": 10.5,  # 궤도 주기
        "pl_trandurh": 3.2,  # 통과 지속시간
        "pl_trandep": 1500,  # 통과 깊이
        "pl_rade": 1.2,  # 행성 반지름
        "pl_insol": 50,  # 일조량
        "pl_eqt": 500,  # 평형 온도
        "st_tmag": 12.5,  # TESS magnitude
        "st_dist": 100,  # 거리
        "st_teff": 5800,  # 별 온도
        "st_logg": 4.5,  # 표면 중력
        "st_rad": 1.0,  # 별 반지름
    }

    try:
        pred, prob = predict_exoplanet_candidate(example_observation)
        print(f"\n예측 결과: {'외계행성 후보' if pred == 1 else '비후보'}")
        print(f"후보 확률: {prob:.2%}")
    except FileNotFoundError as e:
        print(f"\n오류: {e}")


if __name__ == "__main__":
    print("=" * 80)
    print("외계행성 분류 모델 - 예측 예제")
    print("=" * 80)

    # 1. 저장된 모델로 배치 예측
    predict_with_saved_models()

    # 2. 단일 데이터 예측 함수
    # create_prediction_function()

    # 2. 단일 데이터 예측 함수
    # create_prediction_function()
