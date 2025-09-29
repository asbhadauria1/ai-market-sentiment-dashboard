import os
import re
import pandas as pd
from transformers import pipeline

project_root = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(project_root, "data")

sentiment_pipeline = pipeline("sentiment-analysis")


def clean_headlines(df, text_column="title"):
    df = df.copy()

    def clean_text(text):
        text = text.lower()
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    df[text_column] = df[text_column].apply(clean_text)
    return df


def analyze_sentiment(df, text_column="title"):
    df = df.copy()
    df["sentiment"] = df[text_column].apply(lambda x: sentiment_pipeline(x)[0]["label"])
    # Map to numeric score
    mapping = {"POSITIVE": 1, "NEGATIVE": -1, "NEUTRAL": 0}
    df["sentiment_score"] = df["sentiment"].map(mapping).fillna(0)
    return df


if __name__ == "__main__":
    news_files = [f for f in os.listdir(data_dir) if f.endswith("_news.csv")]
    for news_file in news_files:
        ticker = news_file.replace("_news.csv", "")
        df = pd.read_csv(os.path.join(data_dir, news_file))
        df_clean = clean_headlines(df)
        clean_path = os.path.join(data_dir, f"{ticker}_news_clean.csv")
        df_clean.to_csv(clean_path, index=False)
        print(f"Saved cleaned news: {clean_path}")
        df_sentiment = analyze_sentiment(df_clean)
        sentiment_path = os.path.join(data_dir, f"{ticker}_news_sentiment.csv")
        df_sentiment.to_csv(sentiment_path, index=False)
        print(f"Saved sentiment news: {sentiment_path}")
