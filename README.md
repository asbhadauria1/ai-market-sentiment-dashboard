# 📊 AI-Powered Stock Sentiment Dashboard

An end-to-end dashboard that combines financial news sentiment analysis with stock market prediction. This project showcases **Natural Language Processing (NLP)**, **Machine Learning (ML)**, and interactive visualizations using **Streamlit**.

---

## 🚀 Features

### 🔹 Data Enhancements

- **Multi-Ticker Support** → Analyze multiple stocks (AAPL, TSLA, MSFT, etc.)
- **Custom Date Range** → Filter sentiment & stock data by any date window
- **Sentiment Scoring** → Convert news into numeric sentiment (+1, 0, –1)
- **Rolling Features** → 7-day sentiment averages & volatility analysis

### 🔹 Dashboard / UI

- **Interactive Candlestick Charts** (Plotly)
- **Word Cloud** of financial news headlines
- **Sentiment Trendlines** with rolling averages
- **Top Positive & Negative Headlines** section
- **Export Data** → Download CSV/Excel directly
- **Dark/Light Mode** toggle

### 🔹 Machine Learning (ML)

- **Random Forest Classifier** → Predict stock movement (Up/Down)
- **Feature Importance** visualization
- **Explainable AI (SHAP)** → Show which features/headlines influenced predictions
- **Model Confidence Scores** included

### 🔹 Extra Analytics

- **Correlation** between sentiment & stock returns
- **Volume Overlay** vs sentiment
- **Moving Averages** (7-day, 30-day)
- **Sector Comparison** → Analyze multiple stocks side by side

---

## 🛠️ Tech Stack

**Backend / Data:**

- Python, Pandas, NumPy
- yFinance (stock data)
- News API / Alternative APIs (financial headlines)

**NLP & ML:**

- Hugging Face Transformers (BERT sentiment model)
- Scikit-learn (Random Forest, Logistic Regression)
- SHAP (model explainability)

**Frontend / Visualization:**

- Streamlit (interactive dashboard)
- Plotly (candlestick & charts)
- Matplotlib & WordCloud

**Deployment:**

- Streamlit Cloud (free hosting)
- GitHub (version control & portfolio)

---

## ⚡ How It Works

1. **Data Loader** → Fetches stock data via yFinance & financial news
2. **Sentiment Analyzer** → Uses a pre-trained NLP model to classify headlines
3. **Main Script** → Merges stock + sentiment data, adds rolling features, trains ML models
4. **Dashboard** → Streamlit app to visualize, explore, and predict market moves

**Pipeline:**  
`Data Collection → Sentiment Analysis → Feature Engineering → Prediction → Visualization`

---

## 🔧 Installation & Usage

1️⃣ **Clone the Repository**

````bash
git clone https://github.com/asbhadauria1/ai-market-sentiment-dashboard.git
cd ai-market-sentiment-dashboard

---

## ⚡ How It Works

1. **Data Loader** → Fetches stock data via yFinance & financial news
2. **Sentiment Analyzer** → Uses a pre-trained NLP model to classify headlines
3. **Main Script** → Merges stock + sentiment data, adds rolling features, trains ML models
4. **Dashboard** → Streamlit app to visualize, explore, and predict market moves

**Pipeline:**
`Data Collection → Sentiment Analysis → Feature Engineering → Prediction → Visualization`

---

## 🔧 Installation & Usage

1️⃣ **Clone the Repository**
```bash
git clone https://github.com/asbhadauria1/ai-market-sentiment-dashboard.git
cd market-sentiment-dashboard
pip install -r requirements.txt
streamlit run src/dashboard.py --server.headless true

````
