import pandas as pd
from pathlib import Path
raw_dir= Path("data/raw")
raw_dir.mkdir(parents=True, exist_ok=True)
processed_dir= Path("data/processed")
processed_dir.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(raw_dir/"sagd_raw.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values(['well_id','timestamp'])
# 1. SHIFT: 4 steps means 20 minutes
df['target_gas_20min'] = df.groupby('well_id')['future_gas_risk_20min'].shift(-4)
df['target_sand_20min'] = df.groupby('well_id')['future_sand_risk_20min'].shift(-4)
df['target_failure_20min'] = df.groupby('well_id')['failure_probability_20min_ahead'].shift(-4)
# 2. FEATURE ENGINEERING
# Lags
df['sor_lag1'] = df.groupby('well_id')['sor'].shift(1)
df['sor_lag2'] = df.groupby('well_id')['sor'].shift(2)
df['oil_lag1'] = df.groupby('well_id')['oil_rate'].shift(1)
df['pres_lag1'] = df.groupby('well_id')['pressure_kpa'].shift(1)
df['temp_lag1'] = df.groupby('well_id')['temperature_c'].shift(1)
# Rolling Means (6 = 30 min, 12 = 60 min)
df['sor_roll_mean_6'] = df.groupby('well_id')['sor'].transform(lambda x: x.rolling(6, min_periods=1).mean())
df['sor_roll_mean_12'] = df.groupby('well_id')['sor'].transform(lambda x: x.rolling(12, min_periods=1).mean())
df['oil_roll_mean_6'] = df.groupby('well_id')['oil_rate'].transform(lambda x: x.rolling(6, min_periods=1).mean())
df['pres_roll_mean_6'] = df.groupby('well_id')['pressure_kpa'].transform(lambda x: x.rolling(6, min_periods=1).mean())
df['temp_roll_mean_6'] = df.groupby('well_id')['temperature_c'].transform(lambda x: x.rolling(6, min_periods=1).mean())
# Differences
df['sor_diff'] = df['sor'] - df['sor_lag1']
df['oil_diff'] = df['oil_rate'] - df['oil_lag1']
df['pres_diff'] = df['pressure_kpa'] - df['pres_lag1']
df['temp_diff'] = df['temperature_c'] - df['temp_lag1']
df['sor_minus_roll'] = df['sor'] - df['sor_roll_mean_6']
df['oil_minus_roll'] = df['oil_rate'] - df['oil_roll_mean_6']
df['sor_x_temp'] = df['sor'] * df['temperature_c'] / 1000
df['pres_x_oil'] = df['pressure_kpa'] * df['oil_rate'] / 100000
# Time features
df['hour'] = df['timestamp'].dt.hour
df['day_of_run'] = (df['timestamp'] - df.groupby('well_id')['timestamp'].transform('min')).dt.days
# 3.labels
df['is_failure_20min'] = (df['target_failure_20min'] >= 70).astype(int)

df_processed = df.dropna().reset_index(drop=True)
df_processed.to_csv(processed_dir/"sagd_processed.csv", index=False)

print(f"Saved PROCESSED: {df_processed.shape}")
print(f"Failure rate: {df_processed['is_failure_20min'].mean():.4f}")
print(f"Columns ready: {df_processed.columns.tolist()}")