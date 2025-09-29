import os
import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import shap
import warnings

warnings.filterwarnings("ignore")

project_root = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(project_root, "data")
config_path = os.path.join(project_root, "config.json")
with open(config_path, "r") as f:
    config = json.load(f)

tickers = config["default_tickers"]
start_date = pd.to_datetime(config["default_start_date"])
end_date = pd.to_datetime(config["default_end_date"])


def combine_sentiment_stock(news_file, stock_file, combined_file):
    print(f"🔧 Combining sentiment and stock data for {news_file} and {stock_file}")

    # Read news data
    news_df = pd.read_csv(news_file)
    print(f"📰 News data: {len(news_df)} rows")

    if "published" not in news_df.columns or "sentiment_score" not in news_df.columns:
        raise ValueError("News CSV must have 'published' and 'sentiment_score' columns")

    # Fix: Properly parse news dates
    news_df["published"] = pd.to_datetime(news_df["published"], errors="coerce")
    news_df = news_df.dropna(subset=["published"])
    news_df["date_only"] = news_df["published"].dt.date

    print(
        f"📅 News date range: {news_df['date_only'].min()} to {news_df['date_only'].max()}"
    )
    print(f"🎭 Sentiment scores in news: {news_df['sentiment_score'].unique()}")

    # Group by date for sentiment
    sentiment_summary = (
        news_df.groupby("date_only")["sentiment_score"].mean().reset_index()
    )
    sentiment_summary.rename(
        columns={"sentiment_score": "daily_sentiment"}, inplace=True
    )

    print(f"📊 Daily sentiment summary: {len(sentiment_summary)} days")
    print(
        f"📈 Sentiment range in summary: {sentiment_summary['daily_sentiment'].min()} to {sentiment_summary['daily_sentiment'].max()}"
    )

    # Read stock data
    stock_df = pd.read_csv(stock_file)
    print(f"📊 Stock data: {len(stock_df)} rows")

    # FIX: Handle Date column properly
    if "Date" not in stock_df.columns:
        # If no Date column, use the first column that looks like dates
        for col in stock_df.columns:
            if any(keyword in col.lower() for keyword in ["date", "time"]):
                stock_df.rename(columns={col: "Date"}, inplace=True)
                break

    # Parse stock dates
    stock_df["Date"] = pd.to_datetime(stock_df["Date"], errors="coerce")
    stock_df = stock_df.dropna(subset=["Date"])
    stock_df["Date_only"] = stock_df["Date"].dt.date

    print(
        f"📅 Stock date range: {stock_df['Date_only'].min()} to {stock_df['Date_only'].max()}"
    )

    # Merge with sentiment data - CRITICAL FIX
    print("🔄 Merging stock and sentiment data...")
    combined_df = pd.merge(
        stock_df,
        sentiment_summary,
        left_on="Date_only",
        right_on="date_only",
        how="left",  # Keep all stock dates, even if no sentiment data
    )

    print(f"✅ Merged data: {len(combined_df)} rows")
    print(
        f"📊 Rows with sentiment data: {combined_df['daily_sentiment'].notna().sum()}"
    )

    # Fill NaN values with 0 (no sentiment data for that day)
    combined_df["daily_sentiment"].fillna(0, inplace=True)

    print(
        f"🎭 Final sentiment range: {combined_df['daily_sentiment'].min()} to {combined_df['daily_sentiment'].max()}"
    )
    print(
        f"📈 Non-zero sentiment days: {len(combined_df[combined_df['daily_sentiment'] != 0])}"
    )

    # Calculate rolling sentiment and volatility
    combined_df["sentiment_rolling"] = (
        combined_df["daily_sentiment"].rolling(7, min_periods=1).mean()
    )
    combined_df["volatility"] = combined_df["Close"].rolling(7, min_periods=1).std()

    # Save with proper date format
    combined_df.to_csv(combined_file, index=False)
    print(f"💾 Saved combined data to {combined_file}")

    return combined_df


def stock_predictor(df):
    df = df.copy()
    df["price_up"] = np.where(df["Close"].shift(-1) > df["Close"], 1, 0)
    features = ["daily_sentiment", "sentiment_rolling", "volatility", "Volume"]
    df.dropna(subset=features + ["price_up"], inplace=True)
    X = df[features].astype(float)
    y = df["price_up"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Prediction Accuracy: {acc:.2f}")
    print(classification_report(y_test, preds))

    importances = pd.Series(model.feature_importances_, index=features).sort_values(
        ascending=False
    )
    print("\nFeature Importances:\n", importances)

    # SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    X_test_filled = X_test.fillna(0)
    shap.summary_plot(shap_values, X_test_filled, plot_type="bar", show=True)

    # Add confidence to df
    df.loc[X_test.index, "predicted_up_prob"] = probs[:, 1]
    return model, df


for ticker in tickers:
    print(f"\nProcessing {ticker}...")
    news_file = os.path.join(data_dir, f"{ticker}_news_sentiment.csv")
    stock_file = os.path.join(data_dir, f"{ticker}_history.csv")
    combined_file = os.path.join(data_dir, f"{ticker}_combined.csv")

    df_combined = combine_sentiment_stock(news_file, stock_file, combined_file)
    stock_predictor(df_combined)
