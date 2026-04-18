# ABOUTME: Smoke test for pykrx installation.
# ABOUTME: Queries Samsung Electronics (005930) OHLCV for 2024-06-03 to 2024-06-07 and prints shape/head.

from pykrx import stock

df = stock.get_market_ohlcv("20240603", "20240607", "005930")
print("Shape:", df.shape)
print(df.head())
