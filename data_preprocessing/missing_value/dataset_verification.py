import pandas as pd

df = pd.read_csv('datasets/exoplanets.csv')

print('='*80)
print('🔍 최종 데이터셋 검증 (exoplanets.csv)')
print('='*80)

print(f'\n📊 크기: {df.shape[0]:,} 행 × {df.shape[1]} 컬럼')

print(f'\n📋 컬럼 리스트:')
for i, col in enumerate(df.columns, 1):
    print(f'  {i:2d}. {col}')

print(f'\n⚠️ 결측치 (상위 10개):')
missing = df.isna().sum()
missing = missing[missing > 0].sort_values(ascending=False)
for col, count in missing.head(10).items():
    pct = count / len(df) * 100
    print(f'  • {col:<20} {count:>5} ({pct:>5.2f}%)')

print(f'\n총 결측치: {missing.sum():,} ({missing.sum()/(len(df)*len(df.columns))*100:.2f}%)')

print(f'\n🏷️ 레이블 분포:')
for label, count in df['koi_disposition'].value_counts().items():
    pct = count / len(df) * 100
    print(f'  • {label:<20} {count:>5} ({pct:>5.1f}%)')

print(f'\n📍 데이터 출처:')
for source, count in df['data_source'].value_counts().items():
    pct = count / len(df) * 100
    print(f'  • {source:<20} {count:>5} ({pct:>5.1f}%)')

print(f'\n✨ 주요 컬럼 완성도:')
key_cols = ['koi_period', 'koi_prad', 'koi_srad', 'koi_teq', 'koi_smass', 
            'koi_sma', 'koi_steff', 'koi_duration', 'koi_depth']
for col in key_cols:
    completeness = df[col].notna().sum() / len(df) * 100
    print(f'  • {col:<20} {completeness:>5.1f}%')

print('\n' + '='*80)
print('✅ 검증 완료!')
print('='*80)
