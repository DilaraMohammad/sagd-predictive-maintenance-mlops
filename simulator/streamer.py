from pathlib import Path
import pandas as pd
import requests
import time

PROCESSED_FILE = Path(__file__).resolve().parent.parent / "data" / "processed" / "sagd_processed.csv"
API_URL = "http://127.0.0.1:55474"

def stream_data():
    if not PROCESSED_FILE.exists():
        print(f"Error: processed file not found at {PROCESSED_FILE}")
        return

    print("Loading processed data for streaming...")
    df = pd.read_csv(PROCESSED_FILE)
    print(f"Starting stream... Total rows to send: {len(df)}")

    while True:
        for row in df.itertuples(index=False):
            row_dict = row._asdict()
            
            features_dict = {
                k: float(v) for k, v in row_dict.items() 
                if k not in ["well_id", "timestamp", "gas_state", "sand_state", "event_mode"] and pd.notna(v)
            }

            payload = {"features": features_dict}

            try:
                response = requests.post(f"{API_URL}/predict", json=payload)
                if response.status_code == 200:
                    print(f"Row sent successfully! Response: {response.json()}")
                else:
                    print(f"Failed row: Status {response.status_code} | Response: {response.text}")
            except Exception as e:
                print(f"Connection error: {e}")

            time.sleep(3)

if __name__ == "__main__":
    stream_data()