# 모델 정확도: 0.8230


import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from xgboost.callback import TrainingCallback
from tqdm import tqdm

# 데이터 로드 및 전처리
dataset = pd.read_csv("datasets/final_exoplanets.csv")
features = ["pl_rade", "pl_orbper", "pl_trandep", "pl_trandurh", 
            "pl_tranmid", "pl_insol", "pl_eqt", 
            "st_teff", "st_rad", "st_tmag", "st_dist", "st_dens"]
target = "tfopwg_disp"

# 결측치 중앙값 대체
for col in features:
    dataset[col] = dataset[col].fillna(dataset[col].median())

# 파생 변수
dataset['rade_per_orb'] = dataset['pl_rade'] / (dataset['pl_orbper'] + 1e-6)
dataset['eqt_insol_ratio'] = dataset['pl_eqt'] / (dataset['pl_insol'] + 1e-6)
features += ['rade_per_orb', 'eqt_insol_ratio']

# 타겟 인코딩
le = LabelEncoder()
dataset[target] = le.fit_transform(dataset[target])

# 학습/검증 분할
X = dataset[features]
y = dataset[target]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 스케일링
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# DMatrix 변환
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# XGBoost 파라미터 (개선)
params = {
    "objective": "multi:softmax",
    "num_class": len(np.unique(y_train)),
    "learning_rate": 0.01,     # 낮춰서 더 안정적 학습
    "max_depth": 10,           # 깊게 해서 복잡한 패턴 학습
    "subsample": 0.9,          # 과소적합 완화
    "colsample_bytree": 0.9,   # 과소적합 완화
    "reg_alpha": 0.1,          # L1 규제
    "reg_lambda": 1.5,         # L2 규제
    "seed": 42
}

num_rounds = 2000  # 학습률 낮춰서 트리 수 늘림

# TQDM Callback
class TQDMCallback(TrainingCallback):
    def __init__(self, total):
        self.pbar = tqdm(total=total, desc="XGB 학습")

    def after_iteration(self, model, epoch, evals_log):
        self.pbar.n = epoch + 1
        self.pbar.refresh()
        return False

    def after_training(self, model):
        self.pbar.close()
        return model

# 학습
bst = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=num_rounds,
    evals=[(dtest, "eval")],
    verbose_eval=False,
    callbacks=[TQDMCallback(total=num_rounds)]
)

# 예측 및 평가
y_pred = bst.predict(dtest)
print("\nXGBoost 단독 모델 정확도:", accuracy_score(y_test, y_pred))
