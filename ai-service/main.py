"""
ExoVision AI Service - FastAPI Application
외계행성 탐지 API 서비스
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

from app.core.config import settings
from app.schemas.exoplanet import HealthCheckResponse
from app.api.v1.exoplanet import router as exoplanet_router
from app.api.v1.statistics import router as statistics_router
from app.domain.model_loader import model_service

# FastAPI 애플리케이션 생성
app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS 미들웨어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=settings.CORS_ALLOW_METHODS,
    allow_headers=settings.CORS_ALLOW_HEADERS,
)

# 라우터 등록
app.include_router(exoplanet_router)
app.include_router(statistics_router)


@app.get("/")
async def root():
    """
    서비스 정보 엔드포인트
    """
    return {
        "service": settings.API_TITLE,
        "version": settings.API_VERSION,
        "description": settings.API_DESCRIPTION,
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "beginner_predict": "/api/v1/exoplanet/predict/beginner",
            "expert_predict": "/api/v1/exoplanet/predict/expert",
            "mission_stats": "/api/v1/statistics/mission/{mission_name}",
            "all_stats": "/api/v1/statistics/all",
            "combined_stats": "/api/v1/statistics/combined"
        },
        "modes": {
            "beginner": {
                "description": "5 core parameters prediction (others auto-filled)",
                "parameters": ["koi_prad", "dec", "koi_smet", "planet_star_ratio", "planet_density_proxy"]
            },
            "expert": {
                "description": "All 29 parameters input",
                "parameters_count": 29
            }
        },
        "model_info": {
            "model1": "CatBoost Binary Classifier (CONFIRMED vs FALSE POSITIVE)",
            "model2": "Voting Classifier (CANDIDATE detection)",
            "total_parameters": 29,
            "output_classes": ["CONFIRMED", "CANDIDATE", "FALSE POSITIVE"]
        }
    }


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """
    헬스 체크 엔드포인트
    """
    return HealthCheckResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        model_loaded=model_service.is_loaded(),
        service="exovision-ai-service"
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
