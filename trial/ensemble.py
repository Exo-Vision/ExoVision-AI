# 모델 정확도: 0.8274


import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# [1] 데이터 로드
data = pd.read_csv("datasets/final_exoplanets.csv")

target = "tfopwg_disp"  # 실제 타겟 컬럼 이름
X = data.drop(target, axis=1)
y = data[target]

# -------------------------------
# 문자열 타겟 -> 숫자 변환
# -------------------------------
le = LabelEncoder()
y = le.fit_transform(y)

# [2] 스케일링 & 분할
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# [3] 모델 정의
rf = RandomForestClassifier(
    n_estimators=600, max_depth=12, min_samples_split=5, min_samples_leaf=2, random_state=42
)
xgb = XGBClassifier(
    n_estimators=600, learning_rate=0.045, max_depth=10,
    subsample=0.98, colsample_bytree=0.85, min_child_weight=1, gamma=0.05,
    tree_method="hist", random_state=42
)
lgb = LGBMClassifier(
    n_estimators=700, learning_rate=0.05, max_depth=9, subsample=0.95,
    colsample_bytree=0.8, min_child_samples=20, reg_lambda=0.7, random_state=42
)
cb = CatBoostClassifier(
    iterations=700,
    learning_rate=0.04,
    depth=9,
    l2_leaf_reg=3,
    bootstrap_type='Bernoulli',   # subsample 사용 가능
    subsample=0.85,
    verbose=False,
    random_state=42
)


# [4] 학습
rf.fit(X_train, y_train)
xgb.fit(X_train, y_train)
lgb.fit(X_train, y_train)
cb.fit(X_train, y_train)

# [5] 개별 예측
pred_rf = rf.predict_proba(X_test)
pred_xgb = xgb.predict_proba(X_test)
pred_lgb = lgb.predict_proba(X_test)
pred_cb = cb.predict_proba(X_test)

# [6] Weighted Ensemble (XGB 중심)
weights = np.array([0.2, 0.35, 0.3, 0.15])
final_pred = (
    pred_rf * weights[0]
    + pred_xgb * weights[1]
    + pred_lgb * weights[2]
    + pred_cb * weights[3]
)
final_pred_label = np.argmax(final_pred, axis=1)

# [7] 평가
acc_rf = accuracy_score(y_test, np.argmax(pred_rf, axis=1))
acc_xgb = accuracy_score(y_test, np.argmax(pred_xgb, axis=1))
acc_lgb = accuracy_score(y_test, np.argmax(pred_lgb, axis=1))
acc_cb = accuracy_score(y_test, np.argmax(pred_cb, axis=1))
acc_final = accuracy_score(y_test, final_pred_label)

print("\n==================== ✅ 최종 모델 정확도 ====================")
print(f"RF 정확도: {acc_rf:.4f}")
print(f"XGB 정확도: {acc_xgb:.4f}")
print(f"LGB 정확도: {acc_lgb:.4f}")
print(f"CB 정확도: {acc_cb:.4f}")
print(f"\n🚀 Weighted Ensemble 최종 정확도: {acc_final:.4f}")
