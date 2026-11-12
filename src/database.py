import sqlite3
import os
from datetime import datetime

DB_PATH = "logs/nids_telemetry.db"

def init_db():
    os.makedirs("logs", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS telemetry_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            mode TEXT,
            prediction TEXT,
            risk_score REAL,
            risk_level TEXT,
            recommended_action TEXT
        )
    ''')
    conn.commit()
    conn.close()

def log_prediction(mode, prediction, risk_score, risk_level, recommended_action):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    cursor.execute('''
        INSERT INTO telemetry_logs (timestamp, mode, prediction, risk_score, risk_level, recommended_action)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (timestamp, mode, prediction, risk_score, risk_level, recommended_action))
    
    conn.commit()
    conn.close()

def fetch_logs():
    init_db()
    conn = sqlite3.connect(DB_PATH)
    df = pd_read_logs(conn)
    conn.close()
    return df

def pd_read_logs(conn):
    import pandas as pd
    try:
        return pd.read_sql("SELECT * FROM telemetry_logs ORDER BY id DESC", conn)
    except Exception:
        return pd.DataFrame()