import ccxt
import pandas as pd

exchange = ccxt.binance()

def fetch_crypto(symbol="BTC/USDT", timeframe="1d", limit=500):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(
        ohlcv,
        columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["symbol"] = symbol
    return df

if __name__ == "__main__":
    df = fetch_crypto()
    df.to_csv("crypto_raw.csv", index=False)
    print("✅ Données Binance récupérées")
