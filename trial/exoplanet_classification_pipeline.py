"""
외계행성 분류 파이프라인
- TESS 데이터 증강 및 외계행성 후보 분류 모델
- Kepler+K2 통합 데이터로 후보 vs 확정 분류 모델
- Optuna 하이퍼파라미터 최적화
- 모델 저장/로드 기능
"""

import json
import os
import pickle
import warnings
from datetime import datetime

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")


class ExoplanetClassificationPipeline:
    """외계행성 분류 파이프라인 클래스"""

    def __init__(self, data_dir="datasets", models_dir="models"):
        self.data_dir = data_dir
        self.models_dir = models_dir

        # 모델 디렉토리 생성
        os.makedirs(models_dir, exist_ok=True)

        # 데이터 경로
        self.tess_path = os.path.join(data_dir, "tess.csv")
        self.kepler_path = os.path.join(data_dir, "kepler.csv")
        self.k2_path = os.path.join(data_dir, "k2.csv")

        # 모델 저장
        self.tess_model = None
        self.kepler_k2_model = None
        self.tess_best_params = None
        self.kepler_k2_best_params = None

    # ==================== 1. TESS 유용한 컬럼 선택 ====================
    def select_useful_tess_columns(self):
        """TESS 데이터셋에서 외계행성 분류에 유용한 컬럼 선택"""
        print("\n" + "=" * 80)
        print("1단계: TESS 데이터셋에서 유용한 컬럼 선택")
        print("=" * 80)

        # TESS 데이터 로드
        tess_df = pd.read_csv(self.tess_path)
        print(f"원본 TESS 데이터 shape: {tess_df.shape}")
        print(f"전체 컬럼 수: {len(tess_df.columns)}")

        # 분류에 유용한 컬럼 선택
        # 1. Target/Label: tfopwg_disp (처리 상태)
        # 2. Planet properties: 궤도 주기, 반지름, 온도, 일조량 등
        # 3. Transit properties: 지속시간, 깊이 등
        # 4. Stellar properties: 별의 특성

        useful_columns = [
            # Identification
            "toi",
            "tid",
            "tfopwg_disp",
            # Planet Properties (핵심 특성)
            "pl_orbper",  # 궤도 주기
            "pl_trandurh",  # 통과 지속시간
            "pl_trandep",  # 통과 깊이
            "pl_rade",  # 행성 반지름
            "pl_insol",  # 일조량
            "pl_eqt",  # 평형 온도
            "pl_tranmid",  # 통과 중간 시간
            # Stellar Properties (별 특성)
            "st_tmag",  # TESS magnitude
            "st_dist",  # 거리
            "st_teff",  # 별 온도
            "st_logg",  # 표면 중력
            "st_rad",  # 별 반지름
            "st_pmra",  # 적경 고유운동
            "st_pmdec",  # 적위 고유운동
            # Position (ra, dec) 제외 - 외계행성 물리적 특성과 무관
        ]

        # 존재하는 컬럼만 선택
        available_columns = [col for col in useful_columns if col in tess_df.columns]
        tess_filtered = tess_df[available_columns].copy()

        print(f"\n선택된 유용한 컬럼 수: {len(available_columns)}")
        print(f"선택된 컬럼: {available_columns}")
        print(f"필터링된 TESS 데이터 shape: {tess_filtered.shape}")

        # 타겟 레이블 분포 확인
        if "tfopwg_disp" in tess_filtered.columns:
            print("\nTFOPWG Disposition 분포:")
            print(tess_filtered["tfopwg_disp"].value_counts())

        # 저장
        output_path = os.path.join(self.data_dir, "tess_useful_columns.csv")
        tess_filtered.to_csv(output_path, index=False)
        print(f"\n저장 완료: {output_path}")

        return tess_filtered, available_columns

    # ==================== 2. Kepler/K2 데이터를 TESS 형식으로 매핑 ====================
    def map_kepler_k2_to_tess(self):
        """Kepler와 K2 데이터를 TESS 형식으로 매핑"""
        print("\n" + "=" * 80)
        print("2단계: Kepler/K2 데이터를 TESS 형식으로 매핑")
        print("=" * 80)

        # Kepler 데이터 로드
        kepler_df = pd.read_csv(self.kepler_path, low_memory=False)
        print(f"Kepler 데이터 shape: {kepler_df.shape}")

        # K2 데이터 로드 (있는 경우)
        try:
            k2_df = pd.read_csv(self.k2_path, low_memory=False)
            print(f"K2 데이터 shape: {k2_df.shape}")
        except:
            k2_df = None
            print("K2 데이터 로드 실패 또는 파일 없음")

        # Kepler -> TESS 컬럼 매핑
        kepler_to_tess_mapping = {
            # Identification
            "kepid": "tid",
            "koi_disposition": "tfopwg_disp",
            # Planet Properties
            "koi_period": "pl_orbper",
            "koi_duration": "pl_trandurh",  # hours로 변환 필요 (원본은 hours)
            "koi_depth": "pl_trandep",  # ppm 단위
            "koi_prad": "pl_rade",
            "koi_insol": "pl_insol",
            "koi_teq": "pl_eqt",
            "koi_time0bk": "pl_tranmid",
            # Stellar Properties
            "koi_kepmag": "st_tmag",
            "koi_steff": "st_teff",
            "koi_slogg": "st_logg",
            "koi_srad": "st_rad",
            # Position (ra, dec) 제외 - 외계행성 판단과 무관
        }

        # K2 -> TESS 컬럼 매핑 (K2는 다른 컬럼명 사용)
        k2_to_tess_mapping = {
            # Identification
            "epic_hostname": "tid",
            "disposition": "tfopwg_disp",  # K2는 disposition 사용
            # Planet Properties
            "pl_orbper": "pl_orbper",
            "pl_trandur": "pl_trandurh",
            "pl_trandep": "pl_trandep",
            "pl_rade": "pl_rade",
            "pl_insol": "pl_insol",
            "pl_eqt": "pl_eqt",
            "pl_tranmid": "pl_tranmid",
            # Stellar Properties
            "sy_kepmag": "st_tmag",
            "st_teff": "st_teff",
            "st_logg": "st_logg",
            "st_rad": "st_rad",
            # Position (ra, dec) 제외 - 외계행성 판단과 무관
        }

        # Kepler 데이터 매핑
        kepler_mapped = pd.DataFrame()
        for kepler_col, tess_col in kepler_to_tess_mapping.items():
            if kepler_col in kepler_df.columns:
                kepler_mapped[tess_col] = kepler_df[kepler_col]

        # toi 컬럼 생성 (kepid 기반)
        if "kepid" in kepler_df.columns:
            kepler_mapped["toi"] = "K-" + kepler_df["kepid"].astype(str)

        # duration을 hours로 변환 (이미 hours인 경우가 많음)
        # depth를 ppm으로 변환 (Kepler는 이미 ppm)

        # 거리 및 고유운동 추가 (없으면 NaN)
        kepler_mapped["st_dist"] = np.nan
        kepler_mapped["st_pmra"] = np.nan
        kepler_mapped["st_pmdec"] = np.nan

        print(f"\nKepler 매핑 완료: {kepler_mapped.shape}")
        print(f"매핑된 컬럼: {list(kepler_mapped.columns)}")

        # 라벨 통일 (Kepler: CONFIRMED, CANDIDATE, FALSE POSITIVE -> TESS 형식)
        label_mapping = {
            "CONFIRMED": "CP",  # Confirmed Planet
            "CANDIDATE": "PC",  # Planetary Candidate
            "FALSE POSITIVE": "FP",  # False Positive
            "NOT DISPOSITIONED": "APC",  # Ambiguous
        }

        if "tfopwg_disp" in kepler_mapped.columns:
            kepler_mapped["tfopwg_disp"] = kepler_mapped["tfopwg_disp"].map(
                label_mapping
            )
            print("\nKepler 라벨 분포 (변환 후):")
            print(kepler_mapped["tfopwg_disp"].value_counts())

        # K2 데이터도 동일하게 처리
        if k2_df is not None:
            k2_mapped = pd.DataFrame()
            for k2_col, tess_col in k2_to_tess_mapping.items():
                if k2_col in k2_df.columns:
                    k2_mapped[tess_col] = k2_df[k2_col]

            # 필요한 컬럼 추가
            if "epic_hostname" in k2_df.columns:
                k2_mapped["toi"] = "K2-" + k2_df["epic_hostname"].astype(str)
            elif "pl_name" in k2_df.columns:
                k2_mapped["toi"] = "K2-" + k2_df["pl_name"].astype(str)

            k2_mapped["st_dist"] = (
                k2_df["sy_dist"] if "sy_dist" in k2_df.columns else np.nan
            )
            k2_mapped["st_pmra"] = (
                k2_df["sy_pmra"] if "sy_pmra" in k2_df.columns else np.nan
            )
            k2_mapped["st_pmdec"] = (
                k2_df["sy_pmdec"] if "sy_pmdec" in k2_df.columns else np.nan
            )

            if "tfopwg_disp" in k2_mapped.columns:
                k2_mapped["tfopwg_disp"] = k2_mapped["tfopwg_disp"].map(label_mapping)

            print(f"K2 매핑 완료: {k2_mapped.shape}")
        else:
            k2_mapped = None

        # 저장
        kepler_mapped_path = os.path.join(self.data_dir, "kepler_mapped_to_tess.csv")
        kepler_mapped.to_csv(kepler_mapped_path, index=False)
        print(f"\n저장 완료: {kepler_mapped_path}")

        if k2_mapped is not None:
            k2_mapped_path = os.path.join(self.data_dir, "k2_mapped_to_tess.csv")
            k2_mapped.to_csv(k2_mapped_path, index=False)
            print(f"저장 완료: {k2_mapped_path}")

        return kepler_mapped, k2_mapped

    # ==================== 3. TESS 데이터 증강 ====================
    def augment_tess_data(self, tess_df, kepler_mapped, k2_mapped):
        """Kepler와 K2 매핑 데이터로 TESS 증강"""
        print("\n" + "=" * 80)
        print("3단계: TESS 데이터 증강")
        print("=" * 80)

        print(f"원본 TESS shape: {tess_df.shape}")

        # 데이터 결합
        augmented_data = pd.concat([tess_df, kepler_mapped], ignore_index=True)

        if k2_mapped is not None:
            augmented_data = pd.concat([augmented_data, k2_mapped], ignore_index=True)

        print(f"증강된 데이터 shape: {augmented_data.shape}")
        print(
            f"TESS: {len(tess_df)}, Kepler: {len(kepler_mapped)}, K2: {len(k2_mapped) if k2_mapped is not None else 0}"
        )

        # 라벨 분포 확인
        if "tfopwg_disp" in augmented_data.columns:
            print("\n증강된 데이터 라벨 분포:")
            print(augmented_data["tfopwg_disp"].value_counts())

        # 저장
        output_path = os.path.join(self.data_dir, "tess_augmented.csv")
        augmented_data.to_csv(output_path, index=False)
        print(f"\n저장 완료: {output_path}")

        return augmented_data

    # ==================== 4. TESS 모델 학습 (후보 vs 비후보) ====================
    def prepare_tess_data_for_training(self, augmented_data):
        """TESS 학습 데이터 준비 (이진 분류: 외계행성 후보 vs 비후보)"""
        print("\n" + "=" * 80)
        print("4단계: TESS 모델 학습 데이터 준비")
        print("=" * 80)

        # 타겟 레이블 처리
        # PC (Planetary Candidate), CP (Confirmed Planet) -> 1 (후보)
        # FP (False Positive), FA (False Alarm) -> 0 (비후보)

        candidate_labels = ["PC", "CP", "KP"]  # 후보 및 확정
        non_candidate_labels = ["FP", "FA"]  # 비후보

        data = augmented_data.copy()

        # 라벨이 있는 데이터만 사용
        if "tfopwg_disp" not in data.columns:
            raise ValueError("tfopwg_disp 컬럼이 없습니다.")

        # 이진 라벨 생성
        data["is_candidate"] = data["tfopwg_disp"].apply(
            lambda x: (
                1
                if x in candidate_labels
                else (0 if x in non_candidate_labels else np.nan)
            )
        )

        # NaN 제거
        data = data.dropna(subset=["is_candidate"])

        print(f"라벨이 있는 데이터: {len(data)}")
        print(f"클래스 분포:\n{data['is_candidate'].value_counts()}")

        # Feature 선택 (숫자형 컬럼만)
        feature_cols = [
            col
            for col in data.columns
            if col not in ["toi", "tid", "tfopwg_disp", "is_candidate"]
            and data[col].dtype in ["float64", "int64"]
        ]

        X = data[feature_cols].copy()
        y = data["is_candidate"].copy()

        # 결측치 처리 (중앙값으로 대체)
        X = X.fillna(X.median())

        # 무한대 값 처리
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())

        print(f"\nFeature 수: {len(feature_cols)}")
        print(f"Feature 컬럼: {feature_cols}")
        print(f"최종 데이터 shape: X={X.shape}, y={y.shape}")

        return X, y, feature_cols

    def train_tess_model(self, X, y, use_optuna=False, n_trials=50):
        """TESS XGBoost 모델 학습"""
        print("\n" + "=" * 80)
        print("TESS XGBoost 모델 학습")
        print("=" * 80)

        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

        if use_optuna:
            print(f"\nOptuna 하이퍼파라미터 최적화 시작 ({n_trials} trials)...")
            best_params = self.optimize_tess_model(X_train, y_train)
        else:
            # 기본 파라미터
            best_params = {
                "max_depth": 6,
                "learning_rate": 0.1,
                "n_estimators": 100,
                "min_child_weight": 1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "gamma": 0,
                "reg_alpha": 0,
                "reg_lambda": 1,
                "random_state": 42,
                "eval_metric": "logloss",
            }

        self.tess_best_params = best_params

        # 모델 학습
        print("\n모델 학습 중...")
        model = xgb.XGBClassifier(**best_params)
        model.fit(X_train, y_train)

        # 평가
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        test_f1 = f1_score(y_test, y_pred_test)

        print(f"\n=== 모델 성능 ===")
        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test F1 Score: {test_f1:.4f}")

        print("\n=== Classification Report (Test Set) ===")
        print(
            classification_report(
                y_test, y_pred_test, target_names=["Non-Candidate", "Candidate"]
            )
        )

        print("\n=== Confusion Matrix (Test Set) ===")
        cm = confusion_matrix(y_test, y_pred_test)
        print(cm)

        # Feature Importance
        feature_importance = pd.DataFrame(
            {"feature": X.columns, "importance": model.feature_importances_}
        ).sort_values("importance", ascending=False)

        print("\n=== Top 10 Feature Importance ===")
        print(feature_importance.head(10))

        self.tess_model = model

        return model, (X_train, X_test, y_train, y_test)

    def optimize_tess_model(self, X_train, y_train, n_trials=50):
        """Optuna로 TESS 모델 하이퍼파라미터 최적화"""

        def objective(trial):
            params = {
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "gamma": trial.suggest_float("gamma", 0, 5),
                "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
                "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
                "random_state": 42,
                "eval_metric": "logloss",
            }

            model = xgb.XGBClassifier(**params)
            scores = cross_val_score(model, X_train, y_train, cv=3, scoring="f1")
            return scores.mean()

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        print(f"\nBest F1 Score: {study.best_value:.4f}")
        print(f"Best Parameters: {study.best_params}")

        best_params = study.best_params
        best_params["random_state"] = 42
        best_params["eval_metric"] = "logloss"

        return best_params

    # ==================== 5. Kepler + K2 통합 데이터 생성 ====================
    def merge_kepler_k2(self):
        """Kepler와 K2 데이터 통합"""
        print("\n" + "=" * 80)
        print("5단계: Kepler + K2 데이터 통합")
        print("=" * 80)

        # Kepler 데이터 로드
        kepler_df = pd.read_csv(self.kepler_path, low_memory=False)
        print(f"Kepler shape: {kepler_df.shape}")

        # K2 데이터 로드
        try:
            k2_df = pd.read_csv(self.k2_path, low_memory=False)
            print(f"K2 shape: {k2_df.shape}")

            # Kepler는 koi_disposition, K2는 disposition 사용
            # 통일된 컬럼명으로 변경
            if "koi_disposition" in kepler_df.columns:
                kepler_df["disposition"] = kepler_df["koi_disposition"]

            # 공통 컬럼 찾기
            common_cols = list(set(kepler_df.columns) & set(k2_df.columns))
            print(f"\n공통 컬럼 수: {len(common_cols)}")

            # disposition이 공통 컬럼에 있는지 확인
            if (
                "disposition" not in common_cols
                and "koi_disposition" in kepler_df.columns
            ):
                kepler_df_temp = kepler_df.copy()
                kepler_df_temp["disposition"] = kepler_df_temp["koi_disposition"]
                common_cols.append("disposition")
            else:
                kepler_df_temp = kepler_df

            # 공통 컬럼으로 통합
            kepler_common = kepler_df_temp[common_cols].copy()
            k2_common = k2_df[common_cols].copy()

            # 데이터 출처 추가
            kepler_common["data_source"] = "Kepler"
            k2_common["data_source"] = "K2"

            # 통합
            merged_df = pd.concat([kepler_common, k2_common], ignore_index=True)

        except Exception as e:
            print(f"K2 파일 처리 중 오류: {e}")
            print("Kepler만 사용합니다.")
            merged_df = kepler_df.copy()
            if "koi_disposition" in merged_df.columns:
                merged_df["disposition"] = merged_df["koi_disposition"]
            merged_df["data_source"] = "Kepler"

        print(f"\n통합 데이터 shape: {merged_df.shape}")

        # Disposition 분포 확인
        if "disposition" in merged_df.columns:
            print("\nDisposition 분포:")
            print(merged_df["disposition"].value_counts())
        elif "koi_disposition" in merged_df.columns:
            print("\nKOI Disposition 분포:")
            print(merged_df["koi_disposition"].value_counts())

        # 저장
        output_path = os.path.join(self.data_dir, "kepler_k2_merged.csv")
        merged_df.to_csv(output_path, index=False)
        print(f"\n저장 완료: {output_path}")

        return merged_df

    # ==================== 6. Kepler+K2 모델 학습 (후보 vs 확정) ====================
    def prepare_kepler_k2_data_for_training(self, merged_df):
        """Kepler+K2 학습 데이터 준비 (후보 vs 확정)"""
        print("\n" + "=" * 80)
        print("6단계: Kepler+K2 모델 학습 데이터 준비 (후보 vs 확정)")
        print("=" * 80)

        data = merged_df.copy()

        # disposition 컬럼 확인 (koi_disposition 또는 disposition)
        disp_col = None
        if "disposition" in data.columns:
            disp_col = "disposition"
        elif "koi_disposition" in data.columns:
            disp_col = "koi_disposition"
        else:
            raise ValueError("disposition 또는 koi_disposition 컬럼이 없습니다.")

        print(f"사용 컬럼: {disp_col}")

        # CANDIDATE와 CONFIRMED만 사용
        data = data[data[disp_col].isin(["CANDIDATE", "CONFIRMED"])].copy()
        data["is_confirmed"] = (data[disp_col] == "CONFIRMED").astype(int)

        print(f"필터링된 데이터: {len(data)}")
        print(f"클래스 분포:\n{data['is_confirmed'].value_counts()}")

        # Feature 선택
        exclude_cols = [
            "kepid",
            "kepoi_name",
            "kepler_name",
            "koi_disposition",
            "disposition",
            "koi_pdisposition",
            "koi_comment",
            "koi_vet_stat",
            "koi_vet_date",
            "koi_disp_prov",
            "koi_tce_delivname",
            "koi_parm_prov",
            "koi_sparprov",
            "koi_limbdark_mod",
            "koi_trans_mod",
            "koi_fittype",
            "koi_quarters",
            "koi_datalink_dvr",
            "koi_datalink_dvs",
            "data_source",
            "is_confirmed",
            "rowid",
            "pl_name",
            "hostname",
            "pl_letter",
            "k2_name",
            "epic_hostname",
            "epic_candname",
            "hd_name",
            "hip_name",
            "tic_id",
            "gaia_id",
            "default_flag",
            "disp_refname",
            "discoverymethod",
            "disc_year",
            "disc_refname",
            "disc_pubdate",
            "disc_locale",
            "disc_facility",
            "disc_telescope",
            "disc_instrument",
            "soltype",
            "pl_controv_flag",
            "pl_refname",
            "st_refname",
            "sy_refname",
            "rastr",
            "decstr",
            "pl_tsystemref",
            "st_spectype",
            "st_metratio",
            "rowupdate",
            "pl_pubdate",
            "releasedate",
            "k2_campaigns",
        ]

        feature_cols = [
            col
            for col in data.columns
            if col not in exclude_cols and data[col].dtype in ["float64", "int64"]
        ]

        X = data[feature_cols].copy()
        y = data["is_confirmed"].copy()

        # 결측치 및 무한대 처리
        X = X.fillna(X.median())
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())

        print(f"\nFeature 수: {len(feature_cols)}")
        print(f"최종 데이터 shape: X={X.shape}, y={y.shape}")

        return X, y, feature_cols

    def train_kepler_k2_model(self, X, y, use_optuna=False, n_trials=50):
        """Kepler+K2 XGBoost 모델 학습"""
        print("\n" + "=" * 80)
        print("Kepler+K2 XGBoost 모델 학습 (후보 vs 확정)")
        print("=" * 80)

        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

        if use_optuna:
            print(f"\nOptuna 하이퍼파라미터 최적화 시작 ({n_trials} trials)...")
            best_params = self.optimize_kepler_k2_model(X_train, y_train, n_trials)
        else:
            best_params = {
                "max_depth": 6,
                "learning_rate": 0.1,
                "n_estimators": 100,
                "min_child_weight": 1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "gamma": 0,
                "reg_alpha": 0,
                "reg_lambda": 1,
                "random_state": 42,
                "eval_metric": "logloss",
            }

        self.kepler_k2_best_params = best_params

        # 모델 학습
        print("\n모델 학습 중...")
        model = xgb.XGBClassifier(**best_params)
        model.fit(X_train, y_train)

        # 평가
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        test_f1 = f1_score(y_test, y_pred_test)

        print(f"\n=== 모델 성능 ===")
        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test F1 Score: {test_f1:.4f}")

        print("\n=== Classification Report (Test Set) ===")
        print(
            classification_report(
                y_test, y_pred_test, target_names=["Candidate", "Confirmed"]
            )
        )

        print("\n=== Confusion Matrix (Test Set) ===")
        cm = confusion_matrix(y_test, y_pred_test)
        print(cm)

        # Feature Importance
        feature_importance = pd.DataFrame(
            {"feature": X.columns, "importance": model.feature_importances_}
        ).sort_values("importance", ascending=False)

        print("\n=== Top 10 Feature Importance ===")
        print(feature_importance.head(10))

        self.kepler_k2_model = model

        return model, (X_train, X_test, y_train, y_test)

    def optimize_kepler_k2_model(self, X_train, y_train, n_trials=50):
        """Optuna로 Kepler+K2 모델 하이퍼파라미터 최적화"""

        def objective(trial):
            params = {
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "gamma": trial.suggest_float("gamma", 0, 5),
                "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
                "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
                "random_state": 42,
                "eval_metric": "logloss",
            }

            model = xgb.XGBClassifier(**params)
            scores = cross_val_score(model, X_train, y_train, cv=3, scoring="f1")
            return scores.mean()

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        print(f"\nBest F1 Score: {study.best_value:.4f}")
        print(f"Best Parameters: {study.best_params}")

        best_params = study.best_params
        best_params["random_state"] = 42
        best_params["eval_metric"] = "logloss"

        return best_params

    # ==================== 7. 모델 저장/로드 ====================
    def save_models(self, prefix=""):
        """모델과 파라미터 저장"""
        print("\n" + "=" * 80)
        print("모델 저장")
        print("=" * 80)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # TESS 모델 저장
        if self.tess_model is not None:
            tess_model_path = os.path.join(
                self.models_dir, f"{prefix}tess_model_{timestamp}.pkl"
            )
            with open(tess_model_path, "wb") as f:
                pickle.dump(self.tess_model, f)
            print(f"TESS 모델 저장: {tess_model_path}")

            if self.tess_best_params is not None:
                tess_params_path = os.path.join(
                    self.models_dir, f"{prefix}tess_params_{timestamp}.json"
                )
                with open(tess_params_path, "w") as f:
                    json.dump(self.tess_best_params, f, indent=4)
                print(f"TESS 파라미터 저장: {tess_params_path}")

        # Kepler+K2 모델 저장
        if self.kepler_k2_model is not None:
            kepler_model_path = os.path.join(
                self.models_dir, f"{prefix}kepler_k2_model_{timestamp}.pkl"
            )
            with open(kepler_model_path, "wb") as f:
                pickle.dump(self.kepler_k2_model, f)
            print(f"Kepler+K2 모델 저장: {kepler_model_path}")

            if self.kepler_k2_best_params is not None:
                kepler_params_path = os.path.join(
                    self.models_dir, f"{prefix}kepler_k2_params_{timestamp}.json"
                )
                with open(kepler_params_path, "w") as f:
                    json.dump(self.kepler_k2_best_params, f, indent=4)
                print(f"Kepler+K2 파라미터 저장: {kepler_params_path}")

        print("\n모든 모델 저장 완료!")

    def load_models(self, tess_model_path=None, kepler_k2_model_path=None):
        """저장된 모델 로드"""
        print("\n" + "=" * 80)
        print("모델 로드")
        print("=" * 80)

        if tess_model_path:
            with open(tess_model_path, "rb") as f:
                self.tess_model = pickle.load(f)
            print(f"TESS 모델 로드 완료: {tess_model_path}")

        if kepler_k2_model_path:
            with open(kepler_k2_model_path, "rb") as f:
                self.kepler_k2_model = pickle.load(f)
            print(f"Kepler+K2 모델 로드 완료: {kepler_k2_model_path}")

        print("\n모든 모델 로드 완료!")

    # ==================== 전체 파이프라인 실행 ====================
    def run_full_pipeline(self, use_optuna=False, n_trials=50):
        """전체 파이프라인 실행"""
        print("\n" + "=" * 80)
        print("외계행성 분류 전체 파이프라인 시작")
        print("=" * 80)

        # 1. TESS 유용한 컬럼 선택
        tess_filtered, tess_columns = self.select_useful_tess_columns()

        # 2. Kepler/K2를 TESS 형식으로 매핑
        kepler_mapped, k2_mapped = self.map_kepler_k2_to_tess()

        # 3. TESS 데이터 증강
        augmented_tess = self.augment_tess_data(tess_filtered, kepler_mapped, k2_mapped)

        # 4. TESS 모델 학습 (후보 vs 비후보)
        X_tess, y_tess, tess_features = self.prepare_tess_data_for_training(
            augmented_tess
        )
        tess_model, tess_data = self.train_tess_model(
            X_tess, y_tess, use_optuna, n_trials
        )

        # 5. Kepler + K2 통합
        merged_kepler_k2 = self.merge_kepler_k2()

        # 6. Kepler+K2 모델 학습 (후보 vs 확정)
        X_kepler, y_kepler, kepler_features = self.prepare_kepler_k2_data_for_training(
            merged_kepler_k2
        )
        kepler_model, kepler_data = self.train_kepler_k2_model(
            X_kepler, y_kepler, use_optuna, n_trials
        )

        # 7. 모델 저장
        self.save_models()

        print("\n" + "=" * 80)
        print("전체 파이프라인 완료!")
        print("=" * 80)

        return {
            "tess_model": tess_model,
            "kepler_k2_model": kepler_model,
            "tess_data": tess_data,
            "kepler_k2_data": kepler_data,
        }


# ==================== 메인 실행 ====================
if __name__ == "__main__":
    print("=" * 80)
    print("외계행성 분류 파이프라인 시작")
    print("=" * 80)

    # 파이프라인 인스턴스 생성
    pipeline = ExoplanetClassificationPipeline(data_dir="datasets", models_dir="models")

    # 전체 파이프라인 실행
    # use_optuna=True로 설정하면 하이퍼파라미터 최적화 수행
    # n_trials는 Optuna 시도 횟수 (많을수록 오래 걸림)
    results = pipeline.run_full_pipeline(
        use_optuna=True, n_trials=30  # Optuna 최적화 사용  # 30회 시도 (빠른 테스트용)
    )

    print("\n" + "=" * 80)
    print("모든 작업 완료!")
    print("=" * 80)
    print("\n생성된 파일:")
    print("- datasets/tess_useful_columns.csv")
    print("- datasets/kepler_mapped_to_tess.csv")
    print("- datasets/k2_mapped_to_tess.csv")
    print("- datasets/tess_augmented.csv")
    print("- datasets/kepler_k2_merged.csv")
    print("- models/ (학습된 모델 및 파라미터)")
    print("- datasets/tess_augmented.csv")
    print("- datasets/kepler_k2_merged.csv")
    print("- models/ (학습된 모델 및 파라미터)")
