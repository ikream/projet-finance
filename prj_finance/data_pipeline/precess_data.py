import pandas as pd
import numpy as np

def calculate_indicators(df):
    """Ajoute RSI et SMA sur un DataFrame propre"""
    df = df.sort_values('Date') 
    
  
    df['SMA_14'] = df['Close'].rolling(window=14).mean()
   
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    df = df.drop_duplicates()

    return df

def clean_yahoo(file_path):
    print("Traitement du fichier Yahoo...")
    
    try:
        
        df = pd.read_csv(file_path, header=[0, 1], parse_dates=True, index_col=0)
        
        
        df_stacked = df.stack(level=1).reset_index()
        
       
        df_stacked.columns = ['Date', 'Symbol', 'Close', 'High', 'Low', 'Open', 'Volume']
        
        return df_stacked
    except Exception as e:
        print(f"Erreur lecture Yahoo complexe: {e}")
        # Plan B si le format est simple
        return pd.DataFrame()

def clean_binance(file_path):
    print("Traitement du fichier Binance...")
    df = pd.read_csv(file_path)
    
    df['Date'] = pd.to_datetime(df['date'])
    
    df['Symbol'] = df['symbol'].str.replace('/', '-')
    df['Symbol'] = df['Symbol'].str.replace('USDT', 'USD')
    

    cols = ['Date', 'Symbol', 'close', 'high', 'low', 'open', 'volume']
    df = df[cols]
  
    df.columns = ['Date', 'Symbol', 'Close', 'High', 'Low', 'Open', 'Volume']
    
    return df


df_yahoo = clean_yahoo('stocks_raw.csv')
df_binance = clean_binance('crypto_raw.csv') 

print("Fusion des données...")
full_df = pd.concat([df_yahoo, df_binance], ignore_index=True)

print("Calcul des indicateurs techniques...")
full_df = full_df.groupby('Symbol').apply(calculate_indicators).reset_index(drop=True)

output_file = "final_dataset_ready_for_ai.csv"
full_df.to_csv(output_file, index=False)

print(f"✅ Terminé ! Fichier généré : {output_file}")
print(full_df.head())