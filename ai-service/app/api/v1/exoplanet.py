"""
API endpoints for exoplanet detection
"""
from fastapi import APIRouter, HTTPException
from app.schemas.exoplanet import (
    BeginnerExoplanetInput,
    ExpertExoplanetInput,
    ExoplanetPredictionResult
)
from app.domain.model_loader import model_service
from app.core.feature_engineering import (
    create_full_features_from_beginner,
    validate_expert_input
)

router = APIRouter(prefix="/api/v1/exoplanet", tags=["Exoplanet Detection"])


@router.post("/predict/beginner", response_model=ExoplanetPredictionResult)
async def predict_beginner(input_data: BeginnerExoplanetInput):
    """
    초보자 모드: 5개 핵심 파라미터로 외계행성 예측
    
    **입력 파라미터:**
    - koi_prad: 행성 반지름 (Earth radii)
    - dec: 적위 (degrees)
    - koi_smet: 별의 금속성 [Fe/H]
    - planet_star_ratio: 행성/별 크기 비율
    - planet_density_proxy: 행성 밀도 근사값
    
    **나머지 파라미터는 기본값으로 자동 설정됩니다.**
    """
    try:
        # 모델 로드 확인
        if not model_service.is_loaded():
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # 5개 파라미터를 28개로 확장
        input_dict = input_data.model_dump()
        full_features_df = create_full_features_from_beginner(input_dict)
        
        # 예측 수행
        prediction, probability, details = model_service.predict(full_features_df)
        
        return ExoplanetPredictionResult(
            prediction=prediction,
            probability=probability,
            confidence=probability / 100.0,
            classification_details=details
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/predict/expert", response_model=ExoplanetPredictionResult)
async def predict_expert(input_data: ExpertExoplanetInput):
    """
    전문가 모드: 모든 파라미터를 입력하여 외계행성 예측
    
    **총 28개 파라미터:**
    
    **위치 정보:**
    - ra: 적경 (degrees)
    - dec: 적위 (degrees)
    
    **궤도 파라미터:**
    - koi_period: 궤도 주기 (days)
    - koi_eccen: 궤도 이심률
    - koi_longp: 근일점 경도 (degrees)
    - koi_incl: 궤도 기울기 (degrees)
    - koi_impact: 충돌 매개변수
    - koi_sma: 궤도 긴반지름 (AU)
    
    **통과 파라미터:**
    - koi_duration: 통과 지속시간 (hours)
    - koi_depth: 통과 깊이 (ppm)
    - koi_prad: 행성 반지름 (Earth radii)
    
    **행성 환경:**
    - koi_insol: 복사 에너지 (Earth flux)
    - koi_teq: 평형 온도 (K)
    
    **항성 파라미터:**
    - koi_srad: 별 반지름 (Solar radii)
    - koi_smass: 별 질량 (Solar mass)
    - koi_sage: 별 나이 (Gyr)
    - koi_steff: 별 유효 온도 (K)
    - koi_slogg: 별 표면 중력 log10(cm/s²)
    - koi_smet: 별의 금속성 [Fe/H]
    
    **엔지니어링 피처:**
    - orbital_energy: 궤도 에너지
    - transit_signal: 통과 신호 강도
    - stellar_density: 별 밀도
    - log_period: 로그 궤도 주기
    - log_depth: 로그 통과 깊이
    - log_insol: 로그 복사 에너지
    - orbit_stability: 궤도 안정성
    - transit_snr: 통과 신호 대 잡음비
    - planet_star_ratio: 행성/별 크기 비율
    - planet_density_proxy: 행성 밀도 근사값
    """
    try:
        # 모델 로드 확인
        if not model_service.is_loaded():
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # 입력 검증 및 DataFrame 변환
        input_dict = input_data.model_dump()
        full_features_df = validate_expert_input(input_dict)
        
        # 예측 수행
        prediction, probability, details = model_service.predict(full_features_df)
        
        return ExoplanetPredictionResult(
            prediction=prediction,
            probability=probability,
            confidence=probability / 100.0,
            classification_details=details
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
