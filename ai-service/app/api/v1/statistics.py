"""
API endpoints for mission statistics
미션별 통계 데이터 엔드포인트
"""
from fastapi import APIRouter, HTTPException
from typing import Optional
from app.schemas.exoplanet import MissionStatisticsResponse
from app.core.mission_data import get_mission_statistics, get_combined_statistics

router = APIRouter(prefix="/api/v1/statistics", tags=["Mission Statistics"])


@router.get("/mission/{mission_name}", response_model=MissionStatisticsResponse)
async def get_mission_stats(mission_name: str):
    """
    특정 미션의 통계 데이터 조회
    
    **지원 미션:**
    - TESS
    - K2
    - Kepler
    """
    mission_data = get_mission_statistics(mission_name)
    
    if not mission_data:
        raise HTTPException(
            status_code=404, 
            detail=f"Mission '{mission_name}' not found. Available: TESS, K2, Kepler"
        )
    
    return MissionStatisticsResponse(
        mission=mission_name.upper(),
        planet_radius_distribution=mission_data["PlanetRadiusCount"],
        transit_duration_distribution=mission_data["TransitDurationCount"],
        transit_duration_median=mission_data["TransitDurationMedian"]
    )


@router.get("/all")
async def get_all_mission_stats():
    """
    모든 미션의 통계 데이터 조회
    """
    return get_mission_statistics()


@router.get("/combined")
async def get_combined_mission_stats():
    """
    모든 미션의 종합 통계
    - 미션별 중앙값
    - 미션별 총 행성 수
    """
    return get_combined_statistics()
