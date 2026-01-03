import pandas as pd
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus
import sys

# Configuration console
sys.stdout.reconfigure(encoding='utf-8')

file_path = "final_dataset_ready_for_ai.csv"

db_port = 5434

db_password = "password"

connection_string = f"postgresql+pg8000://postgres:{quote_plus(db_password)}@127.0.0.1:{db_port}/finsense_db"

engine = create_engine(connection_string)

df_yahoo_raw = pd.read_csv("stocks_raw.csv")
df_binance_raw = pd.read_csv("crypto_raw.csv")

df_yahoo_raw.to_sql("stocks_raw", con=engine, if_exists="replace", index=False)
df_binance_raw.to_sql("crypto_raw", con=engine, if_exists="replace", index=False)

df_clean = pd.read_csv("final_dataset_ready_for_ai.csv")
df_clean.to_sql("market_data_clean", con=engine, if_exists="replace", index=False)