"""
외계행성 분류 파이프라인 - 빠른 실행 스크립트
기본 파라미터로 빠르게 실행 (Optuna 최적화 없음)
"""

from exoplanet_classification_pipeline import ExoplanetClassificationPipeline

if __name__ == "__main__":
    print("=" * 80)
    print("외계행성 분류 파이프라인 - 빠른 실행 모드")
    print("=" * 80)
    print("\n주의: 이 스크립트는 Optuna 최적화를 사용하지 않습니다.")
    print("최적화를 원하시면 exoplanet_classification_pipeline.py를 실행하세요.\n")

    # 파이프라인 생성
    pipeline = ExoplanetClassificationPipeline(data_dir="datasets", models_dir="models")

    # 전체 파이프라인 실행 (Optuna 최적화 없음 - 빠름)
    results = pipeline.run_full_pipeline(
        use_optuna=False, n_trials=0  # 최적화 비활성화
    )

    print("\n" + "=" * 80)
    print("빠른 실행 완료!")
    print("=" * 80)
    print("\n다음 단계:")
    print("1. models/ 폴더에서 저장된 모델 확인")
    print("2. datasets/ 폴더에서 생성된 데이터 확인")
    print("3. 성능 향상을 원하시면 Optuna 최적화 버전 실행:")
    print("   python exoplanet_classification_pipeline.py")
