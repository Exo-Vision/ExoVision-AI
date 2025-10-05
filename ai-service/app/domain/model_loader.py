"""
Domain layer: Model loading and prediction logic
"""
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple
from app.core.config import settings


class ExoplanetModelService:
    """
    외계행성 탐지 모델 서비스
    - 모델 로딩
    - 예측 수행
    - 결과 해석
    """
    
    def __init__(self):
        self.model1 = None
        self.scaler1 = None
        self.model2 = None
        self.scaler2 = None
        self.config = None
        self._load_models()
    
    def _load_models(self):
        """모델 및 스케일러 로드"""
        try:
            self.model1 = joblib.load(settings.MODEL1_PATH)
            self.scaler1 = joblib.load(settings.SCALER1_PATH)
            self.model2 = joblib.load(settings.MODEL2_PATH)
            self.scaler2 = joblib.load(settings.SCALER2_PATH)
            self.config = joblib.load(settings.CONFIG_PATH)
            
            print("[OK] Models loaded successfully!")
            print(f"  - Model 1: {self.config['model1_name']} ({self.config['model1_accuracy']*100:.2f}%)")
            print(f"  - Model 2: {self.config['model2_name']} ({self.config['model2_accuracy']*100:.2f}%)")
            print(f"  - Final Accuracy: {self.config['final_accuracy']*100:.2f}%")
            print(f"  - Best Threshold: {self.config['best_threshold']:.2f}")
            
        except Exception as e:
            print(f"[ERROR] Model loading failed: {e}")
            raise
    
    def predict(self, X_df: pd.DataFrame) -> Tuple[str, float, Dict[str, Any]]:
        """
        외계행성 예측 수행
        
        Args:
            X_df: 입력 데이터프레임 (모든 피처 포함)
            
        Returns:
            (prediction, probability, details) 튜플
            - prediction: "CONFIRMED", "CANDIDATE", "FALSE POSITIVE"
            - probability: 확률값 (0-100)
            - details: 상세 정보 딕셔너리
        """
        # 스케일링
        X_scaled_1 = self.scaler1.transform(X_df)
        X_scaled_2 = self.scaler2.transform(X_df)
        
        # 모델 1 예측 (이진 분류)
        proba1 = self.model1.predict_proba(X_scaled_1)
        pred1 = self.model1.predict(X_scaled_1)
        
        # 모델 2 예측 (CANDIDATE 판별)
        pred2 = self.model2.predict(X_scaled_2)
        
        # 최종 예측 로직
        max_proba = proba1[0].max()
        threshold = self.config["best_threshold"]
        
        if max_proba >= threshold:
            # 고확신도 → 모델 1 결과 사용
            prediction = "CONFIRMED" if pred1[0] == 1 else "FALSE POSITIVE"
            confidence = max_proba
        else:
            # 저확신도 → 모델 2로 CANDIDATE 판별
            if pred2[0] == 1:
                prediction = "CANDIDATE"
                confidence = 1 - max_proba  # 저확신도를 confidence로 표현
            else:
                prediction = "CONFIRMED" if pred1[0] == 1 else "FALSE POSITIVE"
                confidence = max_proba
        
        # 확률 계산 (백분율)
        probability = round(confidence * 100, 2)
        
        # 상세 정보
        details = {
            "model1_prediction": "CONFIRMED" if pred1[0] == 1 else "FALSE POSITIVE",
            "model1_probability": round(max_proba * 100, 2),
            "model2_prediction": "CANDIDATE" if pred2[0] == 1 else "NOT CANDIDATE",
            "threshold_used": threshold,
            "classification_method": "high_confidence" if max_proba >= threshold else "low_confidence"
        }
        
        return prediction, probability, details
    
    def is_loaded(self) -> bool:
        """모델 로드 여부 확인"""
        return all([
            self.model1 is not None,
            self.scaler1 is not None,
            self.model2 is not None,
            self.scaler2 is not None,
            self.config is not None
        ])


# Singleton instance
model_service = ExoplanetModelService()
