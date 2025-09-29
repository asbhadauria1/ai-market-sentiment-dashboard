import os
import pandas as pd
import feedparser
from datetime import datetime
import yfinance as yf

project_root = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(project_root, "data")
os.makedirs(data_dir, exist_ok=True)


def fetch_yahoo_news(ticker="AAPL"):
    feed_url = f"https://finance.yahoo.com/rss/headline?s={ticker}"
    feed = feedparser.parse(feed_url)
    rows = []
    for e in feed.entries:
        rows.append(
            {
                "title": e.title,
                "link": e.link,
                "published": getattr(e, "published", datetime.utcnow().isoformat()),
            }
        )
    df = pd.DataFrame(rows)
    csv_path = os.path.join(data_dir, f"{ticker}_news.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(df)} news headlines to {csv_path}")
    return df


def fetch_stock_history(ticker="AAPL", period="1y", interval="1d"):
    df = yf.download(
        ticker, period=period, interval=interval, group_by="ticker", auto_adjust=False
    )

    if df.empty:
        print(f"No stock data returned for {ticker}. Skipping.")
        return pd.DataFrame()

    # Reset index
    df.reset_index(inplace=True)

    # Flatten columns if MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[1] if isinstance(col, tuple) else col for col in df.columns]

    # Rename columns to standard names if necessary
    col_map = {c: c for c in df.columns}
    expected_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    for c in expected_cols:
        if c not in df.columns:
            # Some tickers may have slightly different names, try lowercase match
            match = [x for x in df.columns if x.lower() == c.lower()]
            if match:
                col_map[match[0]] = c

    df.rename(columns=col_map, inplace=True)

    # Keep only expected columns - FIXED: Simplified column selection
    available_cols = [c for c in expected_cols if c in df.columns]
    if "Date" in df.columns:
        available_cols = ["Date"] + available_cols

    df = df[available_cols]

    if "Close" not in df.columns:
        print(f"Error: 'Close' column missing for {ticker}. Cannot proceed.")
        return pd.DataFrame()

    # Convert numeric safely
    numeric_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(subset=["Close"], inplace=True)

    csv_path = os.path.join(data_dir, f"{ticker}_history.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(df)} rows of historical prices to {csv_path}")
    return df


if __name__ == "__main__":
    tickers = input("Enter stock tickers separated by comma (e.g., AAPL,TSLA,MSFT): ")
    tickers = [t.strip().upper() for t in tickers.split(",")]
    for ticker in tickers:
        fetch_yahoo_news(ticker)
        fetch_stock_history(ticker)
