import yfinance as yf
import pandas as pd

def fetch_stock(symbol, start="2022-01-01", end="2024-12-31"):
    data = yf.download(symbol, start=start, end=end)
    data.reset_index(inplace=True)
    data["Symbol"] = symbol
    return data

if __name__ == "__main__":
    symbols = ["AAPL", "MSFT"]
    all_data = []

    for s in symbols:
        df = fetch_stock(s)
        all_data.append(df)

    final_df = pd.concat(all_data)
    final_df.to_csv("stocks_raw.csv", index=False)

    print("✅ Données Yahoo Finance récupérées")
