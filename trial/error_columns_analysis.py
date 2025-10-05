"""
데이터셋별 에러 및 제한 플래그 컬럼 분석 및 통합 전략
"""

import pandas as pd
import numpy as np

def analyze_error_columns():
    """에러 컬럼의 의미와 통합 전략 분석"""
    
    print("=" * 100)
    print("외계행성 데이터셋 에러 및 제한 플래그 컬럼 통합 가이드")
    print("=" * 100)
    print()
    
    # 1. 에러 컬럼의 의미
    print("📚 1. 에러 컬럼의 의미")
    print("=" * 100)
    print()
    
    print("🔹 err1 (Upper Uncertainty)")
    print("   - 측정값의 상위 불확실성 (양의 오차)")
    print("   - 실제 값이 측정값보다 클 가능성")
    print("   - 예: pl_orbper = 10.5, pl_orbpererr1 = 0.2 → 실제 값은 10.5 ~ 10.7 사이")
    print()
    
    print("🔹 err2 (Lower Uncertainty)")
    print("   - 측정값의 하위 불확실성 (음의 오차)")
    print("   - 실제 값이 측정값보다 작을 가능성")
    print("   - 예: pl_orbper = 10.5, pl_orbpererr2 = -0.3 → 실제 값은 10.2 ~ 10.5 사이")
    print()
    
    print("🔹 lim (Limit Flag)")
    print("   - 0: 정상 측정값")
    print("   - 1: 상한값 (Upper Limit) - 실제 값이 이보다 작음")
    print("   - -1: 하한값 (Lower Limit) - 실제 값이 이보다 큼")
    print("   - 예: pl_masse = 100, pl_masselim = 1 → 행성 질량은 100 이하")
    print()
    print()
    
    # 2. 데이터셋별 에러 컬럼 구조
    print("📊 2. 데이터셋별 에러 컬럼 구조")
    print("=" * 100)
    print()
    
    datasets = {
        "Kepler": {
            "error_suffix": ["_err1", "_err2"],
            "has_lim": False,
            "example": "koi_period, koi_period_err1, koi_period_err2",
            "note": "비대칭 에러만 제공 (lim 없음)"
        },
        "K2": {
            "error_suffix": ["err1", "err2"],
            "has_lim": True,
            "example": "pl_orbper, pl_orbpererr1, pl_orbpererr2, pl_orbperlim",
            "note": "비대칭 에러 + 제한 플래그 제공"
        },
        "TESS": {
            "error_suffix": ["err1", "err2"],
            "has_lim": True,
            "example": "pl_orbper, pl_orbpererr1, pl_orbpererr2, pl_orbperlim",
            "note": "비대칭 에러 + 제한 플래그 제공"
        }
    }
    
    for dataset, info in datasets.items():
        print(f"🔹 {dataset} 데이터셋")
        print(f"   에러 접미사: {info['error_suffix']}")
        print(f"   제한 플래그: {'있음' if info['has_lim'] else '없음'}")
        print(f"   예시: {info['example']}")
        print(f"   특징: {info['note']}")
        print()
    
    print()
    
    # 3. 통합 전략
    print("🎯 3. 컬럼 통합 전략")
    print("=" * 100)
    print()
    
    strategies = {
        "전략 1: 단순 평균 에러": {
            "설명": "상위/하위 에러의 절대값 평균 사용",
            "적용 대상": "대칭에 가까운 에러, 일반적인 물리량",
            "공식": "error = (|err1| + |err2|) / 2",
            "장점": "단순하고 직관적",
            "단점": "비대칭성 정보 손실",
            "예시": "궤도 주기, 온도, 반지름 등"
        },
        "전략 2: 가중 평균 (신뢰도 기반)": {
            "설명": "에러가 작을수록 높은 가중치 부여",
            "적용 대상": "여러 측정값이 있을 때",
            "공식": "value = Σ(vi / σi²) / Σ(1 / σi²)",
            "장점": "정확한 측정에 더 큰 영향",
            "단점": "복잡한 계산",
            "예시": "여러 논문의 측정값 통합"
        },
        "전략 3: 최대 에러 사용": {
            "설명": "보수적 접근 - 더 큰 에러 사용",
            "적용 대상": "안전성이 중요한 경우",
            "공식": "error = max(|err1|, |err2|)",
            "장점": "보수적, 안전한 추정",
            "단점": "불확실성 과대평가",
            "예시": "행성 거주 가능성 판단"
        },
        "전략 4: 제한 플래그 고려": {
            "설명": "lim 값에 따라 다른 처리",
            "적용 대상": "K2, TESS의 lim 컬럼이 있는 경우",
            "공식": "lim=0: 정상값, lim=±1: 특수 처리",
            "장점": "물리적 의미 보존",
            "단점": "복잡한 로직",
            "예시": "행성 질량 (측정 불가능한 경우)"
        },
        "전략 5: 상위/하위 에러 분리 보존": {
            "설명": "비대칭 에러를 그대로 유지",
            "적용 대상": "정밀한 분석이 필요한 경우",
            "공식": "value, error_upper, error_lower 3개 컬럼 유지",
            "장점": "정보 손실 없음",
            "단점": "컬럼 수 증가",
            "예시": "논문 작성, 상세 분석"
        }
    }
    
    for strategy, details in strategies.items():
        print(f"📌 {strategy}: {details['설명']}")
        print(f"   적용 대상: {details['적용 대상']}")
        print(f"   공식: {details['공식']}")
        print(f"   장점: {details['장점']}")
        print(f"   단점: {details['단점']}")
        print(f"   예시: {details['예시']}")
        print()
    
    print()
    
    # 4. 권장 통합 방법 (컬럼별)
    print("💡 4. 컬럼별 권장 통합 방법")
    print("=" * 100)
    print()
    
    column_recommendations = {
        "궤도 주기 (pl_orbper)": {
            "특성": "매우 정확한 측정, 에러 작음",
            "권장 전략": "전략 1 (단순 평균)",
            "이유": "측정 정확도가 높고 대칭적",
            "코드": "error = (abs(err1) + abs(err2)) / 2"
        },
        "행성 반지름 (pl_rade)": {
            "특성": "중간 정도 정확도",
            "권장 전략": "전략 1 (단순 평균)",
            "이유": "일반적으로 대칭에 가까움",
            "코드": "error = (abs(err1) + abs(err2)) / 2"
        },
        "행성 질량 (pl_masse)": {
            "특성": "측정 어려움, lim 플래그 중요",
            "권장 전략": "전략 4 (제한 플래그 고려)",
            "이유": "상한값만 알려진 경우 많음",
            "코드": "if lim == 1: '상한값', elif lim == -1: '하한값', else: 정상값"
        },
        "궤도 이심률 (pl_orbeccen)": {
            "특성": "0~1 범위, 비대칭 에러",
            "권장 전략": "전략 5 (분리 보존) 또는 전략 3 (최대 에러)",
            "이유": "경계값(0) 근처에서 비대칭",
            "코드": "error = max(abs(err1), abs(err2))"
        },
        "평형 온도 (pl_eqt)": {
            "특성": "계산값, 중간 정확도",
            "권장 전략": "전략 1 (단순 평균)",
            "이유": "일반적으로 대칭적",
            "코드": "error = (abs(err1) + abs(err2)) / 2"
        },
        "항성 온도 (st_teff)": {
            "특성": "스펙트럼 분석, 체계적 오차 가능",
            "권장 전략": "전략 3 (최대 에러)",
            "이유": "보수적 접근 권장",
            "코드": "error = max(abs(err1), abs(err2))"
        },
        "항성 질량 (st_mass)": {
            "특성": "간접 추정, 큰 불확실성",
            "권장 전략": "전략 3 (최대 에러) 또는 전략 5 (분리 보존)",
            "이유": "큰 불확실성, 비대칭 에러",
            "코드": "error = max(abs(err1), abs(err2))"
        }
    }
    
    for column, details in column_recommendations.items():
        print(f"🔹 {column}")
        print(f"   특성: {details['특성']}")
        print(f"   권장 전략: {details['권장 전략']}")
        print(f"   이유: {details['이유']}")
        print(f"   코드: {details['코드']}")
        print()
    
    print()
    
    # 5. 실제 구현 예제
    print("💻 5. Python 구현 예제")
    print("=" * 100)
    print()
    
    print("```python")
    print("import pandas as pd")
    print("import numpy as np")
    print()
    print("def merge_error_columns(df, base_col, strategy='average'):")
    print('    """')
    print("    에러 컬럼을 하나로 통합하는 함수")
    print("    ")
    print("    Parameters:")
    print("    -----------")
    print("    df : DataFrame")
    print("        데이터프레임")
    print("    base_col : str")
    print("        기본 컬럼명 (예: 'pl_orbper')")
    print("    strategy : str")
    print("        'average': 평균 에러")
    print("        'max': 최대 에러")
    print("        'weighted': 가중 평균")
    print("        'separate': 분리 보존")
    print('    """')
    print("    ")
    print("    # 에러 컬럼명 생성")
    print("    err1_col = base_col + 'err1'")
    print("    err2_col = base_col + 'err2'")
    print("    lim_col = base_col + 'lim'")
    print("    ")
    print("    # 컬럼 존재 여부 확인")
    print("    has_err1 = err1_col in df.columns")
    print("    has_err2 = err2_col in df.columns")
    print("    has_lim = lim_col in df.columns")
    print("    ")
    print("    if not (has_err1 and has_err2):")
    print("        return df  # 에러 컬럼 없으면 그대로 반환")
    print("    ")
    print("    # 전략별 처리")
    print("    if strategy == 'average':")
    print("        # 단순 평균 에러")
    print("        df[base_col + '_error'] = (")
    print("            df[err1_col].abs() + df[err2_col].abs()")
    print("        ) / 2")
    print("        ")
    print("    elif strategy == 'max':")
    print("        # 최대 에러")
    print("        df[base_col + '_error'] = np.maximum(")
    print("            df[err1_col].abs(), ")
    print("            df[err2_col].abs()")
    print("        )")
    print("        ")
    print("    elif strategy == 'weighted':")
    print("        # 에러로 가중 평균 (에러가 작을수록 높은 가중치)")
    print("        weights = 1 / (df[err1_col].abs() + df[err2_col].abs() + 1e-10)")
    print("        df[base_col + '_error'] = (")
    print("            df[err1_col].abs() + df[err2_col].abs()")
    print("        ) / 2")
    print("        ")
    print("    elif strategy == 'separate':")
    print("        # 분리 보존")
    print("        df[base_col + '_error_upper'] = df[err1_col].abs()")
    print("        df[base_col + '_error_lower'] = df[err2_col].abs()")
    print("    ")
    print("    # Limit 플래그 처리")
    print("    if has_lim and strategy != 'separate':")
    print("        df[base_col + '_limit_flag'] = df[lim_col]")
    print("        # lim != 0인 경우 특별 표시")
    print("        df.loc[df[lim_col] != 0, base_col + '_note'] = np.where(")
    print("            df.loc[df[lim_col] != 0, lim_col] > 0,")
    print("            'upper_limit',")
    print("            'lower_limit'")
    print("        )")
    print("    ")
    print("    # 원본 에러 컬럼 제거 (선택사항)")
    print("    # df = df.drop([err1_col, err2_col], axis=1)")
    print("    # if has_lim:")
    print("    #     df = df.drop([lim_col], axis=1)")
    print("    ")
    print("    return df")
    print()
    print()
    print("# 사용 예제")
    print("# df_k2 = pd.read_csv('k2.csv')")
    print("# ")
    print("# # 궤도 주기: 평균 에러")
    print("# df_k2 = merge_error_columns(df_k2, 'pl_orbper', strategy='average')")
    print("# ")
    print("# # 행성 질량: 최대 에러 (보수적)")
    print("# df_k2 = merge_error_columns(df_k2, 'pl_masse', strategy='max')")
    print("# ")
    print("# # 이심률: 분리 보존")
    print("# df_k2 = merge_error_columns(df_k2, 'pl_orbeccen', strategy='separate')")
    print("```")
    print()
    print()
    
    # 6. 주의사항
    print("⚠️ 6. 주의사항")
    print("=" * 100)
    print()
    
    warnings = [
        "NaN 값 처리: 에러 컬럼에 NaN이 있을 수 있음 → fillna() 또는 조건부 처리",
        "제한 플래그 해석: lim != 0인 경우 실제 측정값이 아님",
        "비대칭 에러의 의미: err1과 err2의 부호와 크기가 다를 수 있음",
        "단위 일관성: 에러도 본 값과 같은 단위 사용",
        "정보 손실: 에러를 합치면 원본 정보 일부 손실 → 백업 권장",
        "데이터셋별 차이: Kepler는 lim 없음, K2/TESS는 있음",
        "과학적 엄밀성: 논문용 분석은 전략 5 (분리 보존) 권장"
    ]
    
    for i, warning in enumerate(warnings, 1):
        print(f"{i}. {warning}")
    
    print()
    print()
    
    # 7. 최종 권장사항
    print("✅ 7. 최종 권장사항")
    print("=" * 100)
    print()
    
    recommendations = {
        "머신러닝/분류 모델용": {
            "전략": "전략 1 (단순 평균) + 전략 4 (lim 고려)",
            "이유": "단순하고 해석 가능, lim != 0인 데이터는 제외 또는 별도 처리",
            "추가": "에러 컬럼을 feature로 추가 고려"
        },
        "탐색적 데이터 분석(EDA)용": {
            "전략": "전략 1 (단순 평균)",
            "이유": "빠르고 직관적, 시각화에 적합",
            "추가": "에러바 표시 시 평균 에러 사용"
        },
        "과학 논문/상세 분석용": {
            "전략": "전략 5 (분리 보존)",
            "이유": "정보 손실 없음, 비대칭 에러 중요",
            "추가": "에러 전파 계산 필요"
        },
        "프로덕션 시스템용": {
            "전략": "전략 3 (최대 에러)",
            "이유": "보수적 접근, 안전성 우선",
            "추가": "불확실성 높은 예측 회피"
        }
    }
    
    for purpose, details in recommendations.items():
        print(f"🎯 {purpose}")
        print(f"   권장 전략: {details['전략']}")
        print(f"   이유: {details['이유']}")
        print(f"   추가 고려사항: {details['추가']}")
        print()


if __name__ == "__main__":
    analyze_error_columns()
    print("\n✅ 분석 완료!")
