import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
df = pd.read_csv("data/processed/sagd_processed.csv")
drop_cols = [
    "well_id",
    "timestamp",
    "gas_state",
    "sand_state",
    "event_mode",
    "target_gas_20min",
    "target_sand_20min",
    "target_failure_20min",
    "is_failure_20min",
    "failure_probability_20min_ahead",
    "gas_risk",
    "sand_risk",
    "future_gas_risk_20min",
    "future_sand_risk_20min",
    "reservoir_health",
    "gas_accum",
    "sand_accumulation",
]
feature_cols = [col for col in df.columns if col not in drop_cols]
target_col = "is_failure_20min"
X = df[feature_cols]
y = df[target_col]
n = len(df)
train_idx = int(n * 0.60)
val_idx = int(n * 0.80)
X_train, y_train = X.iloc[:train_idx], y.iloc[:train_idx]
X_val, y_val = X.iloc[train_idx:val_idx], y.iloc[train_idx:val_idx]
X_test, y_test = X.iloc[val_idx:], y.iloc[val_idx:]
pos_count = (y_train == 1).sum()
neg_count = (y_train == 0).sum()
scale_pos_weight_val = (neg_count / pos_count) if pos_count > 0 else 1.0
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight_val,
    reg_alpha=0.5,
    reg_lambda=2.0,
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=30,
)
print("Training model with Train and Validation sets...")
model.fit(
    X_train_scaled,
    y_train,
    eval_set=[(X_val_scaled, y_val)],
    verbose=False
)
y_proba = model.predict_proba(X_test_scaled)[:, 1]
threshold = 0.5
y_pred_custom = (y_proba >= threshold).astype(int)
print("\n--- Final Test Set Classification Report ---")
print(classification_report(y_test, y_pred_custom, digits=4, zero_division=0))
joblib.dump(model, "models/sagd_sensor_model.pkl")
joblib.dump(scaler, "models/sagd_scaler.pkl")
joblib.dump(feature_cols, "models/feature_cols.pkl")
print("\nModel, Scaler, and Feature Columns successfully saved to 'models/' directory!")

importances = model.feature_importances_
feature_importance_dict = dict(zip(feature_cols, importances))
sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
for feature, importance in sorted_features:
  print(f"Feature: {feature} --> Importance: {importance:.4f}")