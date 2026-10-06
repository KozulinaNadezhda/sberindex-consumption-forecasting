"""
Модуль автоматического сбора, NLP-классификации и эконометрического 
согласования новостных данных с временными рядами СберИндекса.
"""

import logging
import os
import numpy as np
import pandas as pd
import torch
from transformers import pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def run_nlp_harmonization(model_name: str = "blanchefort/rubert-base-cased-sentiment", alpha: float = 0.60):
    archive_path = "macro_news_archive.csv"
    if not os.path.exists(archive_path):
        raise FileNotFoundError(f"Файл исследовательского корпуса не найден: {archive_path}")

    df_news = pd.read_csv(archive_path)
    logger.info("Loaded %d verified research records from %s", len(df_news), archive_path)

    logger.info("Loading pre-trained NLP Transformer: %s", model_name)
    device = 0 if torch.cuda.is_available() else -1
    classifier = pipeline("sentiment-analysis", model=model_name, device=device, truncation=True)

    headlines = df_news["headline"].tolist()
    logger.info("Executing neural sentiment classification on %d headlines...", len(headlines))
    predictions = classifier(headlines)

    df_news["label"] = [p["label"] for p in predictions]
    df_news["confidence"] = [p["score"] for p in predictions]

    # Полярность: Positive -> (+score), Negative -> (-score), Neutral -> 0
    def calc_polarity(row):
        if row["label"] == "POSITIVE":
            return float(row["confidence"])
        elif row["label"] == "NEGATIVE":
            return -float(row["confidence"])
        return 0.0

    df_news["polarity"] = df_news.apply(calc_polarity, axis=1)
    df_news["date"] = pd.to_datetime(df_news["date"])

    logger.info("Harmonizing daily sentiment to monthly SberIndex frequency (EMA alpha = %.2f)...", alpha)
    
    # Строго диапазон таргета СберИндекса (2023-2024)
    df_filtered = df_news[(df_news["date"] >= "2023-01-01") & (df_news["date"] <= "2024-12-31")].copy()

    df_monthly = df_filtered.groupby(df_filtered["date"].dt.to_period("M")).agg(
        raw_sentiment=("polarity", "mean"),
        news_volume=("headline", "count")
    ).reset_index()

    df_monthly["period"] = df_monthly["date"].dt.to_timestamp()
    
    # Экспоненциальное затухание памяти информационных шоков
    df_monthly["news_sentiment"] = np.round(df_monthly["raw_sentiment"].ewm(alpha=alpha).mean(), 4)

    # Z-score аномалий для опережающего триггера
    mean_s = df_monthly["news_sentiment"].mean()
    std_s = df_monthly["news_sentiment"].std() if df_monthly["news_sentiment"].std() > 0 else 1.0
    df_monthly["sentiment_zscore"] = np.round((df_monthly["news_sentiment"] - mean_s) / std_s, 2)
    df_monthly["is_news_shock"] = (df_monthly["sentiment_zscore"] < -1.5).astype(int)

    output_path = "news_sentiment_monthly.csv"
    df_monthly[["period", "news_sentiment", "sentiment_zscore", "is_news_shock", "news_volume"]].to_csv(output_path, index=False)
    logger.info("Verified monthly sentiment artifact created: %s (%d months covered)", output_path, len(df_monthly))
    return df_monthly


if __name__ == "__main__":
    run_nlp_harmonization()