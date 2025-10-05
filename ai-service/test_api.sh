#!/bin/bash
# ExoVision AI Service API 테스트 스크립트

BASE_URL="http://localhost:8000"

echo "========================================="
echo "ExoVision AI Service API 테스트"
echo "========================================="
echo ""

# 1. Health Check
echo "1. Health Check"
curl -s "${BASE_URL}/health" | python -m json.tool
echo ""
echo ""

# 2. Root Endpoint
echo "2. Service Info (Root)"
curl -s "${BASE_URL}/" | python -m json.tool
echo ""
echo ""

# 3. 초보자 모드 예측
echo "3. 초보자 모드 예측 테스트"
curl -s -X POST "${BASE_URL}/api/v1/exoplanet/predict/beginner" \
  -H "Content-Type: application/json" \
  -d '{
    "koi_prad": 2.5,
    "dec": 45.0,
    "koi_smet": 0.1,
    "planet_star_ratio": 0.05,
    "planet_density_proxy": 1.2
  }' | python -m json.tool
echo ""
echo ""

# 4. 전문가 모드 예측
echo "4. 전문가 모드 예측 테스트"
curl -s -X POST "${BASE_URL}/api/v1/exoplanet/predict/expert" \
  -H "Content-Type: application/json" \
  -d '{
    "ra": 286.591220,
    "dec": 45.0,
    "koi_period": 5.825805,
    "koi_eccen": 0.0,
    "koi_longp": 90.0,
    "koi_incl": 89.0,
    "koi_impact": 0.264301,
    "koi_sma": 0.063769,
    "koi_duration": 3.755,
    "koi_depth": 1344.016115,
    "koi_prad": 2.5,
    "koi_insol": 236.318239,
    "koi_teq": 1091.12293,
    "koi_srad": 1.028,
    "koi_smass": 0.989137,
    "koi_sage": 5.0,
    "koi_steff": 5720.0,
    "koi_slogg": 4.415,
    "koi_smet": 0.1,
    "orbital_energy": 15.681531,
    "transit_signal": 9269.356704,
    "stellar_density": 0.91019,
    "log_period": 1.92071,
    "log_depth": 7.204161,
    "log_insol": 5.469402,
    "orbit_stability": 0.0,
    "transit_snr": 291.805114,
    "planet_star_ratio": 0.05,
    "planet_density_proxy": 1.2
  }' | python -m json.tool
echo ""
echo ""

# 5. TESS 미션 통계
echo "5. TESS 미션 통계"
curl -s "${BASE_URL}/api/v1/statistics/mission/TESS" | python -m json.tool
echo ""
echo ""

# 6. 종합 통계
echo "6. 종합 통계"
curl -s "${BASE_URL}/api/v1/statistics/combined" | python -m json.tool
echo ""
echo ""

echo "========================================="
echo "테스트 완료!"
echo "========================================="
