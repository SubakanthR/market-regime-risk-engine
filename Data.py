import yfinance as yf

df = yf.download("^NSEI", start="2015-01-01", end="2025-01-01")

print(df.head())
print(df.shape)
import os

os.makedirs("data", exist_ok=True)

df.to_csv("data/nifty_50.csv")

print("CSV saved successfully!")