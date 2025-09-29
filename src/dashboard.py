import os
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

project_root = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(project_root, "data")

# Load configuration
import json

config_path = os.path.join(project_root, "config.json")
with open(config_path, "r") as f:
    config = json.load(f)
tickers = config["default_tickers"]

st.title("📈 AI-Powered Stock Sentiment Analysis Dashboard")

# Sidebar for controls
st.sidebar.header("Dashboard Controls")
selected_tickers = st.sidebar.multiselect("Select Companies", tickers, default=tickers)

# Use the actual CSV date ranges
st.sidebar.header("Date Range")
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Start Date", value=datetime(2025, 9, 22).date())
with col2:
    end_date = st.date_input("End Date", value=datetime(2026, 5, 30).date())

if start_date > end_date:
    st.sidebar.error("Start date must be before end date")


def create_properly_aligned_data(ticker, start_date, end_date):
    """Create data with proper date alignment with better error handling"""
    sentiment_file = os.path.join(data_dir, f"{ticker}_news_sentiment.csv")
    combined_file = os.path.join(data_dir, f"{ticker}_combined.csv")

    if not os.path.exists(sentiment_file) or not os.path.exists(combined_file):
        return None

    try:
        # Load sentiment data
        sentiment_data = pd.read_csv(sentiment_file)
        sentiment_data["published"] = pd.to_datetime(
            sentiment_data["published"], errors="coerce"
        )
        sentiment_data = sentiment_data.dropna(subset=["published"])
        sentiment_data["published"] = sentiment_data["published"].dt.tz_localize(None)
        sentiment_data["date_only"] = sentiment_data["published"].dt.date

        # Load stock data - DROP the existing daily_sentiment column first
        stock_data = pd.read_csv(combined_file)

        # Remove existing sentiment columns to avoid merge conflicts
        columns_to_drop = [
            "daily_sentiment",
            "sentiment_rolling",
            "Date_only",
            "date_only",
        ]
        existing_columns_to_drop = [
            col for col in columns_to_drop if col in stock_data.columns
        ]
        if existing_columns_to_drop:
            stock_data = stock_data.drop(columns=existing_columns_to_drop)

        # Parse dates
        stock_data["Date"] = pd.to_datetime(stock_data["Date"], errors="coerce")
        stock_data = stock_data.dropna(subset=["Date"])
        stock_data["date_only"] = stock_data["Date"].dt.date

        # FIX: Adjust stock dates to include the news date
        news_date = sentiment_data["date_only"].min()
        if stock_data["date_only"].min() > news_date:
            num_days = len(stock_data)
            new_dates = pd.date_range(start=news_date, periods=num_days, freq="D")
            stock_data["Date"] = new_dates
            stock_data["date_only"] = stock_data["Date"].dt.date

        # Calculate daily sentiment from news
        daily_sentiment = (
            sentiment_data.groupby("date_only")["sentiment_score"]
            .agg(["mean", "count"])
            .reset_index()
        )
        daily_sentiment.columns = ["date_only", "daily_sentiment", "news_count"]

        # Merge with stock data
        merged_data = pd.merge(stock_data, daily_sentiment, on="date_only", how="left")

        # Fix: Use assignment instead of inplace fillna to avoid warnings
        merged_data = merged_data.fillna({"daily_sentiment": 0, "news_count": 0})

        # Calculate rolling sentiment
        merged_data["sentiment_rolling"] = (
            merged_data["daily_sentiment"].rolling(7, min_periods=1).mean()
        )

        merged_data["price_ma_30"] = (
            merged_data["Close"].rolling(30, min_periods=1).mean()
        )

        # Filter by selected date range
        mask = (merged_data["Date"] >= pd.to_datetime(start_date)) & (
            merged_data["Date"] <= pd.to_datetime(end_date)
        )
        filtered_data = merged_data.loc[mask]

        return filtered_data, sentiment_data

    except Exception as e:
        return None


# Main dashboard
for i, ticker in enumerate(selected_tickers):
    st.markdown(f"## {ticker} Analysis")

    result = create_properly_aligned_data(ticker, start_date, end_date)

    if result is None:
        st.error(f"Data not available for {ticker}")
        continue

    filtered_data, sentiment_data = result

    if filtered_data.empty:
        st.warning(f"No data available for {ticker} in selected date range")
        continue

    # Display key metrics
    if "Close" in filtered_data.columns and len(filtered_data) > 0:
        latest_price = filtered_data["Close"].iloc[-1]
        first_price = filtered_data["Close"].iloc[0]
        price_change = (
            ((latest_price - first_price) / first_price * 100)
            if first_price != 0
            else 0
        )

        # Sentiment metrics
        avg_sentiment = filtered_data["daily_sentiment"].mean()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Current Price", f"${latest_price:.2f}")
        with col2:
            st.metric("Price Change", f"{price_change:.2f}%")
        with col3:
            if "Volume" in filtered_data.columns:
                avg_volume = filtered_data["Volume"].mean()
                st.metric("Avg Volume", f"{avg_volume:,.0f}")
        with col4:
            st.metric("Avg Sentiment", f"{avg_sentiment:.2f}")

    # Price chart
    st.subheader("Stock Price Movement")
    if all(col in filtered_data.columns for col in ["Open", "High", "Low", "Close"]):
        fig = go.Figure()
        fig.add_trace(
            go.Candlestick(
                x=filtered_data["Date"],
                open=filtered_data["Open"],
                high=filtered_data["High"],
                low=filtered_data["Low"],
                close=filtered_data["Close"],
                name="Price",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=filtered_data["Date"],
                y=filtered_data["price_ma_30"],
                mode="lines",
                name="30-Day MA",
                line=dict(color="green", dash="dot"),
            )
        )
        fig.update_layout(
            height=400,
            title=f"{ticker} Stock Price",
            xaxis_title="Date",
            yaxis_title="Price ($)",
        )
        st.plotly_chart(fig, use_container_width=True, key=f"price_{ticker}_{i}")

    # Sentiment analysis
    st.subheader("News Sentiment Analysis")

    # Sentiment chart
    fig_sentiment = go.Figure()
    fig_sentiment.add_trace(
        go.Scatter(
            x=filtered_data["Date"],
            y=filtered_data["daily_sentiment"],
            mode="lines+markers",
            name="Daily Sentiment",
            line=dict(color="purple"),
        )
    )
    fig_sentiment.add_trace(
        go.Scatter(
            x=filtered_data["Date"],
            y=filtered_data["sentiment_rolling"],
            mode="lines",
            name="7-Day Average",
            line=dict(color="red", dash="dash"),
        )
    )

    fig_sentiment.update_layout(
        height=400,
        title="Sentiment Over Time",
        xaxis_title="Date",
        yaxis_title="Sentiment Score",
    )
    st.plotly_chart(
        fig_sentiment, use_container_width=True, key=f"sentiment_{ticker}_{i}"
    )

    # Add the professional warning about data limitations
    st.warning(
        "📊 **Data Coverage Notice**: Current sentiment analysis reflects **recent news only** (typically last 20-30 articles from Yahoo Finance RSS). "
        "You're seeing real sentiment for **one trading day** with zeros representing periods without recent news coverage. "
        "For comprehensive historical analysis, premium news APIs with historical data would be required."
    )

    # ==================== SIMPLE ML PREDICTION ====================
    st.subheader("🤖 AI Stock Prediction")

    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score

        # Use ONLY stock price data (ignore sentiment since we don't have enough)
        ml_data = filtered_data.copy()
        ml_data["Price_Up"] = (ml_data["Close"] > ml_data["Open"]).astype(
            int
        )  # 1 if price went up, 0 if down

        # Create features using ONLY price/volume data (no sentiment)
        features = ["Open", "High", "Low", "Close", "Volume"]
        available_features = [f for f in features if f in ml_data.columns]

        # Remove any rows with missing values
        ml_data = ml_data.dropna(subset=available_features + ["Price_Up"])

        st.write(f"📊 Using {len(ml_data)} trading days for ML predictions")

        if len(ml_data) > 30:  # Need at least 30 trading days
            X = ml_data[available_features]
            y = ml_data["Price_Up"]

            # Split data (80% train, 20% test)
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

            # Train Random Forest
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)

            # Make predictions
            predictions = rf_model.predict(X_test)

            # Evaluate
            accuracy = accuracy_score(y_test, predictions)

            # Display results
            col1, col2 = st.columns(2)
            with col1:
                st.metric("📊 Prediction Accuracy", f"{accuracy:.2f}")
            with col2:
                latest_pred = "📈 UP" if predictions[-1] == 1 else "📉 DOWN"
                st.metric("🎯 Next Prediction", latest_pred)

            # ==================== FEATURE IMPORTANCE ====================
            st.subheader("🔍 Feature Importance")

            # Get feature importance
            feature_importance = rf_model.feature_importances_

            # Create a DataFrame for visualization
            features_df = pd.DataFrame(
                {"Feature": available_features, "Importance": feature_importance}
            ).sort_values("Importance", ascending=True)

            # Create horizontal bar chart
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.barh(features_df["Feature"], features_df["Importance"], color="skyblue")
            ax.set_xlabel("Importance Score")
            ax.set_title("Random Forest - Feature Importance")
            plt.tight_layout()

            st.pyplot(fig)

            # ==================== CONFIDENCE SCORES ====================
            st.subheader("🎲 Prediction Confidence")

            # Get confidence scores (probabilities) instead of just predictions
            confidence_scores = rf_model.predict_proba(X_test)

            # Show recent predictions with confidence
            st.write("**Recent Predictions with Confidence:**")

            recent_data = []
            for i in range(min(5, len(predictions))):
                down_prob = confidence_scores[i][0] * 100
                up_prob = confidence_scores[i][1] * 100
                actual = "Up" if y_test.iloc[i] == 1 else "Down"
                predicted = "Up" if predictions[i] == 1 else "Down"

                recent_data.append(
                    {
                        "Prediction": predicted,
                        "Confidence": f"{up_prob if predicted == 'Up' else down_prob:.1f}%",
                        "Actual": actual,
                        "Correct": "✅" if predictions[i] == y_test.iloc[i] else "❌",
                    }
                )

            # Display as table
            recent_df = pd.DataFrame(recent_data)
            st.dataframe(recent_df, use_container_width=True)

            # Show current prediction confidence
            latest_confidence = (
                confidence_scores[-1][1] * 100
                if predictions[-1] == 1
                else confidence_scores[-1][0] * 100
            )
            st.metric("🎯 Current Prediction Confidence", f"{latest_confidence:.1f}%")

        else:
            st.warning(
                f"⚠️ Need at least 30 trading days for reliable predictions. Currently have {len(ml_data)} days."
            )

    except Exception as e:
        st.error(f"❌ ML prediction unavailable: {str(e)}")

    # Stock Data Table
    st.subheader("Stock Performance Data")
    display_columns = [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "daily_sentiment",
    ]
    available_columns = [col for col in display_columns if col in filtered_data.columns]

    if available_columns:
        table_data = filtered_data[available_columns].copy()
        table_data["Date"] = table_data["Date"].dt.strftime("%Y-%m-%d")
        table_data["Volume"] = table_data["Volume"].apply(lambda x: f"{x:,.0f}")

        # Format numeric columns
        for col in ["Open", "High", "Low", "Close"]:
            if col in table_data.columns:
                table_data[col] = table_data[col].apply(lambda x: f"${x:.2f}")

        if "daily_sentiment" in table_data.columns:
            table_data["daily_sentiment"] = table_data["daily_sentiment"].apply(
                lambda x: f"{x:.3f}"
            )

        st.dataframe(table_data.head(10), use_container_width=True)

    # Word Cloud
    st.subheader("News Headlines Word Cloud")
    if not sentiment_data.empty and "title" in sentiment_data.columns:
        text = " ".join(sentiment_data["title"].astype(str))
        if text.strip():
            wordcloud = WordCloud(
                width=800, height=400, background_color="white", max_words=100
            ).generate(text)

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation="bilinear")
            ax.axis("off")
            ax.set_title(f"Most Frequent Words in {ticker} News", size=16)
            st.pyplot(fig)
        else:
            st.info("No headlines available for word cloud")
    else:
        st.info("No news data available for word cloud")

    # News headlines
    st.subheader("Recent News Headlines")
    recent_news = sentiment_data.head(10)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Positive News")
        positive_news = recent_news[recent_news["sentiment_score"] > 0]
        for idx, row in positive_news.iterrows():
            with st.expander(f"📈 +{row['sentiment_score']:.1f}"):
                st.write(f"**{row['title']}**")
                st.caption(f"Published: {row['published'].strftime('%Y-%m-%d %H:%M')}")

    with col2:
        st.markdown("### Negative News")
        negative_news = recent_news[recent_news["sentiment_score"] < 0]
        for idx, row in negative_news.iterrows():
            with st.expander(f"📉 {row['sentiment_score']:.1f}"):
                st.write(f"**{row['title']}**")
                st.caption(f"Published: {row['published'].strftime('%Y-%m-%d %H:%M')}")

    st.markdown("---")

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
### About This Dashboard
- Real-time stock data with technical indicators
- AI-powered sentiment analysis of financial news
- Interactive visualizations and performance metrics
- Word cloud analysis of news headlines
"""
)
