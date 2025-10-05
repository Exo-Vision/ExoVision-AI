"""
외계행성 분류 모델 - 예측 전용 스크립트
저장된 모델을 로드하여 새로운 데이터에 대해 예측 수행

필수 입력값 (Features):
====================

🌟 TESS 모델 (외계행성 후보 vs 비후보 분류)
-----------------------------------------
필수 입력 컬럼 14개:

1. pl_orbper      : 궤도 주기 (days) - 행성이 별 주위를 한 바퀴 도는 시간
2. pl_trandurh    : 통과 지속시간 (hours) - 행성이 별 앞을 지나가는 시간
3. pl_trandep     : 통과 깊이 (ppm) - 별 밝기 감소량 (parts per million)
4. pl_rade        : 행성 반지름 (지구 반지름) - 지구=1
5. pl_insol       : 일조량 (지구 대비) - 지구가 받는 태양 에너지=1
6. pl_eqt         : 평형 온도 (Kelvin) - 행성의 이론적 온도
7. pl_tranmid     : 통과 중간 시간 (BJD) - Barycentric Julian Date
8. st_tmag        : TESS magnitude - 별의 밝기
9. st_dist        : 거리 (parsec) - 별까지의 거리
10. st_teff       : 별 유효 온도 (Kelvin)
11. st_logg       : 별 표면 중력 (log10(cm/s²))
12. st_rad        : 별 반지름 (태양 반지름) - 태양=1
13. st_pmra       : 적경 고유운동 (mas/yr) - 별의 움직임
14. st_pmdec      : 적위 고유운동 (mas/yr) - 별의 움직임

⚠️  제거된 컬럼 (외계행성 판단과 무관):
- ra (적경) : 하늘 좌표 - 물리적 특성 아님
- dec (적위) : 하늘 좌표 - 물리적 특성 아님

예측 결과:
- 0: 비후보 (False Positive) - 외계행성이 아님
- 1: 후보 (Candidate/Confirmed) - 외계행성 가능성 높음


🪐 Kepler+K2 모델 (외계행성 후보 → 확정 분류)
------------------------------------------
현재는 공통 컬럼 부족으로 성능 제한적
improve_kepler_k2_model.py 실행 권장
"""

import json
import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd


class ExoplanetPredictor:
    """저장된 외계행성 분류 모델 로드 및 예측"""

    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        self.tess_model = None
        self.kepler_k2_model = None
        self.tess_params = None
        self.kepler_k2_params = None

        # TESS 모델 필수 컬럼 (RA/Dec 제외 - 외계행성 물리적 특성과 무관)
        self.tess_features = [
            "pl_orbper",
            "pl_trandurh",
            "pl_trandep",
            "pl_rade",
            "pl_insol",
            "pl_eqt",
            "pl_tranmid",
            "st_tmag",
            "st_dist",
            "st_teff",
            "st_logg",
            "st_rad",
            "st_pmra",
            "st_pmdec",
        ]

        # Kepler+K2 모델 필수 컬럼 (현재 버전 - 개선 필요)
        self.kepler_k2_features = []  # 개선 후 업데이트 필요

    def load_latest_models(self):
        """가장 최신 모델 자동 로드"""
        import glob

        tess_models = glob.glob(os.path.join(self.models_dir, "tess_model_*.pkl"))
        kepler_models = glob.glob(
            os.path.join(self.models_dir, "kepler_k2_model_*.pkl")
        )

        if tess_models:
            latest_tess = max(tess_models, key=os.path.getctime)
            self.load_tess_model(latest_tess)
            print(f"✅ TESS 모델 로드: {os.path.basename(latest_tess)}")
        else:
            print("⚠️  TESS 모델을 찾을 수 없습니다.")

        if kepler_models:
            latest_kepler = max(kepler_models, key=os.path.getctime)
            self.load_kepler_k2_model(latest_kepler)
            print(f"✅ Kepler+K2 모델 로드: {os.path.basename(latest_kepler)}")
        else:
            print("⚠️  Kepler+K2 모델을 찾을 수 없습니다.")

    def load_tess_model(self, model_path):
        """TESS 모델 로드"""
        with open(model_path, "rb") as f:
            self.tess_model = pickle.load(f)

        # 파라미터 로드 (있는 경우)
        param_path = model_path.replace(".pkl", ".json").replace("_model_", "_params_")
        if os.path.exists(param_path):
            with open(param_path, "r") as f:
                self.tess_params = json.load(f)

    def load_kepler_k2_model(self, model_path):
        """Kepler+K2 모델 로드"""
        with open(model_path, "rb") as f:
            self.kepler_k2_model = pickle.load(f)

        # 파라미터 로드 (있는 경우)
        param_path = model_path.replace(".pkl", ".json").replace("_model_", "_params_")
        if os.path.exists(param_path):
            with open(param_path, "r") as f:
                self.kepler_k2_params = json.load(f)

    def predict_tess(self, data, return_proba=True):
        """
        TESS 모델로 외계행성 후보 예측

        Parameters:
        -----------
        data : pd.DataFrame or dict or list of dict
            예측할 데이터
            필수 컬럼: pl_orbper, pl_trandurh, pl_trandep, pl_rade, pl_insol,
                      pl_eqt, pl_tranmid, st_tmag, st_dist, st_teff, st_logg,
                      st_rad, st_pmra, st_pmdec, ra, dec

        return_proba : bool, default=True
            True: 확률 반환, False: 클래스 레이블만 반환

        Returns:
        --------
        predictions : np.ndarray
            예측 결과 (0: 비후보, 1: 후보)
        probabilities : np.ndarray (return_proba=True인 경우)
            후보일 확률 [비후보 확률, 후보 확률]
        """
        if self.tess_model is None:
            raise ValueError(
                "TESS 모델이 로드되지 않았습니다. load_latest_models()를 먼저 실행하세요."
            )

        # 데이터 변환
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data.copy()

        # 필수 컬럼 확인
        missing_cols = set(self.tess_features) - set(df.columns)
        if missing_cols:
            raise ValueError(f"필수 컬럼 누락: {missing_cols}")

        # Feature 준비
        X = df[self.tess_features].copy()

        # 결측치 처리
        X = X.fillna(0)
        X = X.replace([np.inf, -np.inf], 0)

        # 예측
        predictions = self.tess_model.predict(X)

        if return_proba:
            probabilities = self.tess_model.predict_proba(X)
            return predictions, probabilities
        else:
            return predictions

    def predict_kepler_k2(self, data, return_proba=True):
        """
        Kepler+K2 모델로 후보 → 확정 예측

        Parameters:
        -----------
        data : pd.DataFrame or dict or list of dict
            예측할 데이터
            필수 컬럼: ra, dec (현재 버전)

        return_proba : bool, default=True
            True: 확률 반환, False: 클래스 레이블만 반환

        Returns:
        --------
        predictions : np.ndarray
            예측 결과 (0: 후보, 1: 확정)
        probabilities : np.ndarray (return_proba=True인 경우)
            확정일 확률 [후보 확률, 확정 확률]
        """
        if self.kepler_k2_model is None:
            raise ValueError(
                "Kepler+K2 모델이 로드되지 않았습니다. load_latest_models()를 먼저 실행하세요."
            )

        # 데이터 변환
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data.copy()

        # 필수 컬럼 확인
        missing_cols = set(self.kepler_k2_features) - set(df.columns)
        if missing_cols:
            raise ValueError(f"필수 컬럼 누락: {missing_cols}")

        # Feature 준비
        X = df[self.kepler_k2_features].copy()

        # 결측치 처리
        X = X.fillna(0)
        X = X.replace([np.inf, -np.inf], 0)

        # 예측
        predictions = self.kepler_k2_model.predict(X)

        if return_proba:
            probabilities = self.kepler_k2_model.predict_proba(X)
            return predictions, probabilities
        else:
            return predictions

    def show_required_features(self):
        """필수 입력 컬럼 목록 출력"""
        print("\n" + "=" * 80)
        print("🌟 TESS 모델 필수 입력 (14개 - RA/Dec 제외)")
        print("=" * 80)
        for i, feature in enumerate(self.tess_features, 1):
            print(f"{i:2d}. {feature}")

        print("\n" + "=" * 80)
        print("🪐 Kepler+K2 모델 - 개선 필요")
        print("=" * 80)
        print("improve_kepler_k2_model.py를 실행하여 개선된 모델 사용을 권장합니다.")
        print("=" * 80)


# ========================================
# 사용 예제
# ========================================


def example_single_prediction():
    """단일 관측 데이터 예측 예제"""
    print("\n" + "=" * 80)
    print("예제 1: 단일 관측 데이터 예측")
    print("=" * 80)

    # 예측기 생성 및 모델 로드
    predictor = ExoplanetPredictor()
    predictor.load_latest_models()

    # 예제 관측 데이터 (TESS - RA/Dec 제외)
    observation = {
        "pl_orbper": 10.5,  # 궤도 주기 (일)
        "pl_trandurh": 3.2,  # 통과 지속시간 (시간)
        "pl_trandep": 1500,  # 통과 깊이 (ppm)
        "pl_rade": 1.2,  # 행성 반지름 (지구=1)
        "pl_insol": 50,  # 일조량
        "pl_eqt": 500,  # 평형 온도 (K)
        "pl_tranmid": 2459000,  # 통과 중간 시간
        "st_tmag": 12.5,  # TESS magnitude
        "st_dist": 100,  # 거리 (pc)
        "st_teff": 5800,  # 별 온도 (K)
        "st_logg": 4.5,  # 표면 중력
        "st_rad": 1.0,  # 별 반지름
        "st_pmra": -5.0,  # 고유운동 RA
        "st_pmdec": -3.0,  # 고유운동 Dec
    }

    # 예측
    pred, proba = predictor.predict_tess(observation)

    print(f"\n📊 예측 결과:")
    print(f"  분류: {'✅ 외계행성 후보' if pred[0] == 1 else '❌ 비후보'}")
    print(f"  후보 확률: {proba[0][1]:.2%}")
    print(f"  비후보 확률: {proba[0][0]:.2%}")


def example_batch_prediction():
    """배치 예측 예제 (CSV 파일)"""
    print("\n" + "=" * 80)
    print("예제 2: CSV 파일 배치 예측")
    print("=" * 80)

    # 예측기 생성 및 모델 로드
    predictor = ExoplanetPredictor()
    predictor.load_latest_models()

    # 실제 TESS 데이터 로드 (샘플)
    try:
        data = pd.read_csv("datasets/tess.csv")

        # 처음 10개만 예측
        sample_data = data.head(10)

        # 예측
        predictions, probabilities = predictor.predict_tess(sample_data)

        # 결과 출력
        print(f"\n📊 {len(predictions)}개 샘플 예측 완료\n")

        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            status = "✅ 후보" if pred == 1 else "❌ 비후보"
            confidence = prob[1] if pred == 1 else prob[0]
            print(f"샘플 {i+1}: {status} (확신도: {confidence:.2%})")

        # 결과를 DataFrame으로 저장
        results = pd.DataFrame(
            {
                "prediction": predictions,
                "candidate_probability": probabilities[:, 1],
                "non_candidate_probability": probabilities[:, 0],
            }
        )

        output_path = "predictions_output.csv"
        results.to_csv(output_path, index=False)
        print(f"\n💾 예측 결과 저장: {output_path}")

    except FileNotFoundError:
        print("⚠️  datasets/tess.csv 파일을 찾을 수 없습니다.")


def example_create_sample_input():
    """샘플 입력 CSV 파일 생성"""
    print("\n" + "=" * 80)
    print("예제 3: 샘플 입력 파일 생성")
    print("=" * 80)

    # 샘플 데이터 생성
    sample_data = pd.DataFrame(
        [
            {
                "pl_orbper": 10.5,
                "pl_trandurh": 3.2,
                "pl_trandep": 1500,
                "pl_rade": 1.2,
                "pl_insol": 50,
                "pl_eqt": 500,
                "pl_tranmid": 2459000,
                "st_tmag": 12.5,
                "st_dist": 100,
                "st_teff": 5800,
                "st_logg": 4.5,
                "st_rad": 1.0,
                "st_pmra": -5.0,
                "st_pmdec": -3.0,
                "ra": 123.456,
                "dec": 45.678,
            },
            {
                "pl_orbper": 5.2,
                "pl_trandurh": 2.1,
                "pl_trandep": 800,
                "pl_rade": 0.9,
                "pl_insol": 100,
                "pl_eqt": 600,
                "pl_tranmid": 2459100,
                "st_tmag": 11.2,
                "st_dist": 80,
                "st_teff": 6000,
                "st_logg": 4.3,
                "st_rad": 1.1,
                "st_pmra": -3.0,
                "st_pmdec": -2.0,
                "ra": 200.123,
                "dec": -30.456,
            },
        ]
    )

    output_path = "sample_input.csv"
    sample_data.to_csv(output_path, index=False)
    print(f"✅ 샘플 입력 파일 생성: {output_path}")
    print("\n이 파일을 수정하여 예측에 사용할 수 있습니다:")
    print(
        f"  predictions, probas = predictor.predict_tess(pd.read_csv('{output_path}'))"
    )


def show_feature_descriptions():
    """Feature 설명 출력"""
    print("\n" + "=" * 80)
    print("📖 TESS 모델 입력 변수 상세 설명")
    print("=" * 80)

    descriptions = {
        "pl_orbper": "궤도 주기 (days) - 행성이 별 주위를 한 바퀴 도는데 걸리는 시간",
        "pl_trandurh": "통과 지속시간 (hours) - 행성이 별 앞을 지나가는 시간",
        "pl_trandep": "통과 깊이 (ppm) - 행성이 별을 가릴 때 밝기 감소량",
        "pl_rade": "행성 반지름 (지구 반지름) - 지구를 1로 했을 때 행성 크기",
        "pl_insol": "일조량 (지구=1) - 행성이 받는 별빛의 양",
        "pl_eqt": "평형 온도 (Kelvin) - 행성의 이론적 온도",
        "pl_tranmid": "통과 중간 시간 (BJD) - 관측 시간 (Barycentric Julian Date)",
        "st_tmag": "TESS magnitude - 별의 밝기 (작을수록 밝음)",
        "st_dist": "거리 (parsec) - 별까지의 거리 (1pc ≈ 3.26 광년)",
        "st_teff": "별 유효 온도 (Kelvin) - 별 표면 온도",
        "st_logg": "별 표면 중력 (log g) - 별 표면의 중력 (log 단위)",
        "st_rad": "별 반지름 (태양=1) - 태양을 1로 했을 때 별 크기",
        "st_pmra": "적경 고유운동 (mas/yr) - 별의 하늘에서의 움직임 (동서)",
        "st_pmdec": "적위 고유운동 (mas/yr) - 별의 하늘에서의 움직임 (남북)",
        "ra": "적경 (degrees) - 하늘에서의 위치 (경도 같은 개념)",
        "dec": "적위 (degrees) - 하늘에서의 위치 (위도 같은 개념)",
    }

    for i, (feature, desc) in enumerate(descriptions.items(), 1):
        print(f"\n{i:2d}. {feature}")
        print(f"    {desc}")

    print("\n" + "=" * 80)
    print("💡 팁:")
    print("  - 필수 컬럼이 없으면 0 또는 평균값으로 채워집니다")
    print("  - 단위를 정확히 지켜야 좋은 예측 결과를 얻을 수 있습니다")
    print("  - 실제 관측 데이터에 가까울수록 정확도가 높아집니다")
    print("=" * 80)


# ========================================
# 메인 실행
# ========================================

if __name__ == "__main__":
    print("=" * 80)
    print("🌟 외계행성 분류 모델 - 예측 전용 스크립트")
    print("=" * 80)

    # 1. 필수 입력 변수 확인
    predictor = ExoplanetPredictor()
    predictor.show_required_features()

    # 2. Feature 상세 설명
    show_feature_descriptions()

    # 3. 모델 로드
    print("\n" + "=" * 80)
    print("모델 로드 중...")
    print("=" * 80)
    predictor.load_latest_models()

    # 4. 예제 실행
    try:
        example_single_prediction()
        example_batch_prediction()
        example_create_sample_input()
    except Exception as e:
        print(f"\n⚠️  예제 실행 중 오류: {e}")

    print("\n" + "=" * 80)
    print("✅ 완료!")
    print("=" * 80)
    print("\n💡 사용 방법:")
    print("1. predictor = ExoplanetPredictor()")
    print("2. predictor.load_latest_models()")
    print("3. predictions, probabilities = predictor.predict_tess(your_data)")
    print("\n자세한 내용은 위의 예제 코드를 참고하세요.")
    print("\n자세한 내용은 위의 예제 코드를 참고하세요.")
