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
    try:
        # FIXED: Use proper yfinance syntax
        stock_data = yf.Ticker(ticker)
        df = stock_data.history(period=period, interval=interval)

        if df.empty:
            print(f"No stock data returned for {ticker}. Skipping.")
            return pd.DataFrame()

        # Reset index to convert Date from index to column
        df = df.reset_index()

        # Ensure Date column exists
        if "Date" not in df.columns:
            print(
                f"❌ Date column not found for {ticker}. Available columns: {list(df.columns)}"
            )
            return pd.DataFrame()

        # Select and order columns
        preferred_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
        available_cols = [col for col in preferred_cols if col in df.columns]
        df = df[available_cols]

        # Convert numeric columns
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Remove rows with missing Close prices
        initial_count = len(df)
        df = df.dropna(subset=["Close"])
        final_count = len(df)

        if final_count < initial_count:
            print(
                f"⚠️ Removed {initial_count - final_count} rows with missing Close prices"
            )

        # Ensure Date is proper datetime
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])

        # Sort by date
        df = df.sort_values("Date").reset_index(drop=True)

        csv_path = os.path.join(data_dir, f"{ticker}_history.csv")
        df.to_csv(csv_path, index=False)
        print(f"✅ Saved {len(df)} rows of historical prices to {csv_path}")
        print(f"📅 Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")

        return df

    except Exception as e:
        print(f"❌ Error fetching stock data for {ticker}: {e}")
        return pd.DataFrame()


if __name__ == "__main__":
    tickers = input("Enter stock tickers separated by comma (e.g., AAPL,TSLA,MSFT): ")
    tickers = [t.strip().upper() for t in tickers.split(",")]
    for ticker in tickers:
        print(f"\nProcessing {ticker}...")
        fetch_yahoo_news(ticker)
        fetch_stock_history(ticker)
