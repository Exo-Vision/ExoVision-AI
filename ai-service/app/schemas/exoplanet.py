"""
Pydantic schemas for ExoVision AI Service
"""
from pydantic import BaseModel, Field
from typing import Optional, List


class BeginnerExoplanetInput(BaseModel):
    """
    초보자 모드: 5개 핵심 파라미터
    나머지는 기본값으로 자동 처리
    """
    koi_prad: float = Field(..., description="행성 반지름 (Earth radii)", gt=0)
    dec: float = Field(..., description="적위 (degrees)", ge=-90, le=90)
    koi_smet: float = Field(..., description="별의 금속성 [Fe/H]")
    planet_star_ratio: float = Field(..., description="행성/별 크기 비율", gt=0)
    planet_density_proxy: float = Field(..., description="행성 밀도 근사값", gt=0)


class ExpertExoplanetInput(BaseModel):
    """
    전문가 모드: 모든 파라미터 입력
    총 28개 파라미터 (original 19 + engineered 9)
    """
    # 위치 정보
    ra: float = Field(..., description="적경 (degrees)", ge=0, le=360)
    dec: float = Field(..., description="적위 (degrees)", ge=-90, le=90)
    
    # 궤도 파라미터
    koi_period: float = Field(..., description="궤도 주기 (days)", gt=0)
    koi_eccen: float = Field(..., description="궤도 이심률", ge=0, le=1)
    koi_longp: float = Field(..., description="근일점 경도 (degrees)", ge=0, le=360)
    koi_incl: float = Field(..., description="궤도 기울기 (degrees)", ge=0, le=90)
    koi_impact: float = Field(..., description="충돌 매개변수", ge=0)
    koi_sma: float = Field(..., description="궤도 긴반지름 (AU)", gt=0)
    
    # 통과 파라미터
    koi_duration: float = Field(..., description="통과 지속시간 (hours)", gt=0)
    koi_depth: float = Field(..., description="통과 깊이 (ppm)", gt=0)
    koi_prad: float = Field(..., description="행성 반지름 (Earth radii)", gt=0)
    
    # 행성 환경
    koi_insol: float = Field(..., description="복사 에너지 (Earth flux)", gt=0)
    koi_teq: float = Field(..., description="평형 온도 (K)", gt=0)
    
    # 항성 파라미터
    koi_srad: float = Field(..., description="별 반지름 (Solar radii)", gt=0)
    koi_smass: float = Field(..., description="별 질량 (Solar mass)", gt=0)
    koi_sage: float = Field(..., description="별 나이 (Gyr)", gt=0)
    koi_steff: float = Field(..., description="별 유효 온도 (K)", gt=0)
    koi_slogg: float = Field(..., description="별 표면 중력 log10(cm/s²)", ge=0)
    koi_smet: float = Field(..., description="별의 금속성 [Fe/H]")
    
    # 엔지니어링 피처
    orbital_energy: float = Field(..., description="궤도 에너지")
    transit_signal: float = Field(..., description="통과 신호 강도")
    stellar_density: float = Field(..., description="별 밀도")
    log_period: float = Field(..., description="로그 궤도 주기")
    log_depth: float = Field(..., description="로그 통과 깊이")
    log_insol: float = Field(..., description="로그 복사 에너지")
    orbit_stability: float = Field(..., description="궤도 안정성")
    transit_snr: float = Field(..., description="통과 신호 대 잡음비", gt=0)
    planet_star_ratio: float = Field(..., description="행성/별 크기 비율", gt=0)
    planet_density_proxy: float = Field(..., description="행성 밀도 근사값", gt=0)


class ExoplanetPredictionResult(BaseModel):
    """
    예측 결과 응답
    """
    prediction: str = Field(..., description="예측 결과: CONFIRMED, CANDIDATE, FALSE POSITIVE")
    probability: float = Field(..., description="확률 (0-100%)")
    confidence: float = Field(..., description="신뢰도 (0-1)")
    classification_details: dict = Field(..., description="분류 상세 정보")
    mission_statistics: Optional[dict] = Field(None, description="미션별 통계 데이터")


class MissionStatisticsResponse(BaseModel):
    """
    미션 통계 응답
    """
    mission: str
    planet_radius_distribution: dict
    transit_duration_distribution: dict
    transit_duration_median: float


class HealthCheckResponse(BaseModel):
    """
    Health check 응답
    """
    status: str
    timestamp: str
    model_loaded: bool
    service: str
