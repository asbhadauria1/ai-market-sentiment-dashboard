import os
import pandas as pd
import json

project_root = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(project_root, "data")

# Check all ticker files
tickers = ["AAPL", "TSLA", "MSFT"]  # Add your tickers

print("🔍 DEBUGGING SENTIMENT DATA ISSUES")
print("=" * 50)

for ticker in tickers:
    print(f"\n📊 Analyzing {ticker}...")

    sentiment_file = os.path.join(data_dir, f"{ticker}_news_sentiment.csv")
    combined_file = os.path.join(data_dir, f"{ticker}_combined.csv")
    news_file = os.path.join(data_dir, f"{ticker}_news_clean.csv")

    # Check sentiment file
    if os.path.exists(sentiment_file):
        df_sentiment = pd.read_csv(sentiment_file)
        print(f"✅ Sentiment file found: {len(df_sentiment)} rows")
        print(f"   Columns: {list(df_sentiment.columns)}")

        if "sentiment_score" in df_sentiment.columns:
            print(f"   Sentiment scores: {df_sentiment['sentiment_score'].unique()}")
            print(
                f"   Score range: {df_sentiment['sentiment_score'].min()} to {df_sentiment['sentiment_score'].max()}"
            )
            print(
                f"   Non-zero scores: {len(df_sentiment[df_sentiment['sentiment_score'] != 0])}"
            )
        else:
            print("❌ 'sentiment_score' column missing!")

        if "sentiment" in df_sentiment.columns:
            print(f"   Sentiment labels: {df_sentiment['sentiment'].unique()}")

    # Check combined file
    if os.path.exists(combined_file):
        df_combined = pd.read_csv(combined_file)
        print(f"✅ Combined file found: {len(df_combined)} rows")
        print(f"   Columns: {list(df_combined.columns)}")

        sentiment_cols = [
            col for col in df_combined.columns if "sentiment" in col.lower()
        ]
        if sentiment_cols:
            for col in sentiment_cols:
                print(f"   {col}: {df_combined[col].unique()[:5]}")  # First 5 values
        else:
            print("❌ No sentiment columns in combined file!")

    # Check dates
    if os.path.exists(sentiment_file) and os.path.exists(combined_file):
        df_sentiment = pd.read_csv(sentiment_file)
        df_combined = pd.read_csv(combined_file)

        if "published" in df_sentiment.columns:
            df_sentiment["published"] = pd.to_datetime(
                df_sentiment["published"], errors="coerce"
            )
            print(
                f"   Sentiment dates: {df_sentiment['published'].min()} to {df_sentiment['published'].max()}"
            )

        if "Date" in df_combined.columns:
            df_combined["Date"] = pd.to_datetime(df_combined["Date"], errors="coerce")
            print(
                f"   Combined dates: {df_combined['Date'].min()} to {df_combined['Date'].max()}"
            )

    print("-" * 30)

print("\n🎯 SUMMARY OF ISSUES:")
print("1. If sentiment scores are all 0 → Problem in sentiment_analyzer.py")
print(
    "2. If sentiment columns missing in combined file → Problem in main.py merge logic"
)
print("3. If dates don't match → Problem in date handling")
