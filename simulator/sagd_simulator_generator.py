import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
class SAGDWell:
    def __init__(self, well_id: str):
        self.well_id = well_id
        self.base_sor = np.random.uniform(2.5, 3.0)
        self.base_oil = np.random.uniform(100, 115)
        self.base_pressure = np.random.uniform(1900, 2100)
        self.base_temp = np.random.uniform(208, 218)
        self.reservoir_health = 100.0
        self.sand_accumulation = np.random.uniform(0, 1)
        self.gas_accum = np.random.uniform(0, 1)
        self.event_mode = "NORMAL"
        self.event_timer = 0
        self.event_intensity = 0.0
        self.gas_state = "NORMAL"
        self.sand_state = "NORMAL"

    def update_physics(self):
        self.reservoir_health = max(65.0, self.reservoir_health - np.random.uniform(0.00012, 0.0004))
        self.sand_accumulation = np.clip(self.sand_accumulation + np.random.uniform(0.00003, 0.0004), 0, 25)
        self.gas_accum = np.clip(self.gas_accum * 0.9997 + np.random.uniform(0, 0.0008), 0, 25)
        if self.event_mode == "NORMAL":
            if np.random.random() < 0.0004:
                self.event_mode = np.random.choice(["GAS_BUILDUP", "SAND_BUILDUP"], p=[0.5, 0.5])
                self.event_timer = np.random.randint(50, 140)
                self.event_intensity = np.random.uniform(0.75, 1.0)
        else:
            self.event_timer -= 1
            if self.event_timer <= 0:
                self.event_mode = "NORMAL"
                self.event_intensity = 0.0

    def step(self, current_time: datetime):
        self.update_physics()
        hf = (100.0 - self.reservoir_health) / 100.0
        gas_e = 0.0; sand_e = 0.0
        if self.event_mode == "GAS_BUILDUP":
            prog = 1.0 - (self.event_timer / 140.0)
            gas_e = self.event_intensity * prog * np.random.uniform(0.9, 1.1)
            self.gas_accum += gas_e * 0.10
        elif self.event_mode == "SAND_BUILDUP":
            prog = 1.0 - (self.event_timer / 140.0)
            sand_e = self.event_intensity * prog * np.random.uniform(0.9, 1.1)
            self.sand_accumulation += sand_e * 0.07

        gas_f = self.gas_accum / 20.0 + gas_e
        sand_f = self.sand_accumulation / 25.0 + sand_e

        sor = np.clip(self.base_sor + hf*0.4 + gas_f*1.05 + sand_f*0.3 + np.random.normal(0,0.05), 2.0, 6.5)
        oil = np.clip(self.base_oil - hf*12 - gas_f*5 - sand_f*13 + np.random.normal(0,1.3), 45, 125)
        pres = np.clip(self.base_pressure + gas_f*70 + sand_f*85 - hf*40 + np.random.normal(0,12), 1350, 3100)
        temp = np.clip(self.base_temp + gas_f*7.5 + hf*2.2 + sand_f*0.6 + np.random.normal(0,0.8), 190, 270)

        sor_diff = sor - self.base_sor
        oil_drop = self.base_oil - oil
        pres_diff = pres - self.base_pressure
        temp_diff = temp - self.base_temp

        gas_risk = np.clip((sor_diff*18 + temp_diff*4.2),0,100)
        sand_risk = np.clip((oil_drop*1.8 + pres_diff*0.14),0,100)

        if sor_diff > 1.15 and temp_diff > 5.8:
            self.gas_state = "GAS_INTERFERENCE"
        elif sor_diff > 0.6 or temp_diff > 4.0:
            self.gas_state = "GAS_WARNING"
        else:
            self.gas_state = "NORMAL"

        if oil_drop > 18 and pres_diff > 95:
            self.sand_state = "SAND_INFLUX"
        elif oil_drop > 15 or pres_diff > 80:
            self.sand_state = "SAND_WARNING"
        else:
            self.sand_state = "NORMAL"

        future_gas = np.clip(gas_risk + gas_e*20,0,100)
        future_sand = np.clip(sand_risk + sand_e*20,0,100)

        return {
            "well_id": self.well_id,
            "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "gas_state": self.gas_state,
            "sand_state": self.sand_state,
            "gas_risk": round(float(gas_risk),2),
            "sand_risk": round(float(sand_risk),2),
            "future_gas_risk_20min": round(float(future_gas),2),
            "future_sand_risk_20min": round(float(future_sand),2),
            "failure_probability_20min_ahead": round(float(max(future_gas, future_sand)),2),
            "sor": round(float(sor),3),
            "oil_rate": round(float(oil),1),
            "pressure_kpa": round(float(pres),1),
            "temperature_c": round(float(temp),1),
            "reservoir_health": round(float(self.reservoir_health),2),
            "sand_accumulation": round(float(self.sand_accumulation),3),
            "gas_accum": round(float(self.gas_accum),3),
            "event_mode": self.event_mode,
        }
raw_dir= Path("data/raw")
raw_dir.mkdir(parents=True, exist_ok=True)
field = [SAGDWell(f"SAGD-{i+1:02d}") for i in range(10)]
rows=[]
steps = int(90*24*60/5)
base_time= datetime.now() - timedelta(days=90)

for step_idx in range(steps):
    cur = base_time + timedelta(minutes= 5 * step_idx)
    for w in field:
        rows.append(w.step(cur))

df = pd.DataFrame(rows)
df.to_csv(raw_dir/"sagd_raw.csv", index=False)
print(f"Saved RAW: {df.shape}")