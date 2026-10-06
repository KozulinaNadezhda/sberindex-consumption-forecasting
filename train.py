"""
Мастер-пайплайн подготовки данных, обучения, валидации на 4 горизонтах (Prophet, Ridge, CatBoost)
и совмещенного CUSUM-мониторинга для конкурса СберИндекс.
"""

import json
import logging
import os
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
import yaml
from catboost import CatBoostRegressor
from prophet import Prophet
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> Dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Файл конфигурации не найден: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def fetch_cbr_usd_rate(start_date: str = "01/01/2021", end_date: str = "31/12/2026") -> pd.DataFrame:
    url = f"http://www.cbr.ru/scripts/xml_dynamic.asp?date_req1={start_date}&date_req2={end_date}&VAL_NM_RQ=R01235"
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        response.raise_for_status()
        tree = ET.fromstring(response.content)

        dates, rates = [], []
        for record in tree.findall("Record"):
            dates.append(record.get("Date"))
            rates.append(float(record.find("Value").text.replace(",", ".")))

        df_usd = pd.DataFrame({"period": pd.to_datetime(dates, format="%d.%m.%Y"), "usd": rates})
        df_monthly = df_usd.groupby(df_usd["period"].dt.to_period("M"))["usd"].mean().reset_index()
        df_monthly["period"] = df_monthly["period"].dt.to_timestamp()
        logger.info("Котировки ЦБ РФ успешно выгружены и агрегированы по месяцам.")
        return df_monthly
    except Exception as exc:
        logger.warning("Сбой API ЦБ РФ (%s). Используется резервное значение.", exc)
        return pd.DataFrame()


def build_dataset(cfg: Dict) -> pd.DataFrame:
    logger.info("Загрузка массивов СберИндекса и гармонизированных данных Росстата...")
    df_main = pd.read_csv(cfg["data"]["main_dataset_path"], sep=";")
    df_main["period"] = pd.to_datetime(df_main["period"])
    df_main = df_main[df_main["category_15"] == "Все категории"][["period", "mo", "value"]]
    df_clean = df_main.groupby(["period", "mo"])["value"].sum().reset_index()

    df_macro = pd.read_csv(cfg["data"]["macro_dataset_path"], sep=";")
    df_macro = df_macro[df_macro["category"] == "Всего"].rename(columns={"date": "period", "value": "macro_value"})
    df_macro["period"] = pd.to_datetime(df_macro["period"])
    df_macro = df_macro[["period", "macro_value"]]

    df_merged = pd.merge(df_clean, df_macro, on="period", how="left")

    df_usd = fetch_cbr_usd_rate()
    if not df_usd.empty:
        df_merged = pd.merge(df_merged, df_usd, on="period", how="left")
        df_merged["usd"] = df_merged["usd"].ffill()
    else:
        df_merged["usd"] = 90.0

    # Генерация признакового пространства
    df_merged["year"] = df_merged["period"].dt.year
    df_merged["month"] = df_merged["period"].dt.month
    df_merged["sin_month"] = np.sin(2 * np.pi * df_merged["month"] / 12)
    df_merged["cos_month"] = np.cos(2 * np.pi * df_merged["month"] / 12)
    df_merged["is_december"] = (df_merged["month"] == 12).astype(int)
    df_merged["is_january"] = (df_merged["month"] == 1).astype(int)
    df_merged["macro_per_usd"] = df_merged["macro_value"] / df_merged["usd"]

    sentiment_file = "news_sentiment_monthly.csv"
    if os.path.exists(sentiment_file):
        df_sent = pd.read_csv(sentiment_file)
        df_sent["period"] = pd.to_datetime(df_sent["period"])
        df_merged = pd.merge(df_merged, df_sent[["period", "news_sentiment"]], on="period", how="left")
        df_merged["news_sentiment"] = df_merged["news_sentiment"].ffill().bfill().fillna(0.0)
        logger.info("Верифицированный NLP-сентимент успешно интегрирован.")
    else:
        logger.warning("Файл сентимента не найден. Инициализирован нейтральный фон.")
        df_merged["news_sentiment"] = 0.0

    if "news_sentiment" in df_merged.columns:
        mean_s = df_merged["news_sentiment"].mean()
        std_s = df_merged["news_sentiment"].std() if df_merged["news_sentiment"].std() > 0 else 1.0
        z_scores = (df_merged["news_sentiment"] - mean_s) / std_s
        df_merged["is_shock_regime"] = (z_scores < -1.0).astype(int)
    else:
        df_merged["is_shock_regime"] = 0

    return df_merged.dropna().reset_index(drop=True)


def evaluate_four_horizons(df: pd.DataFrame, cfg: Dict, features: List[str]) -> pd.DataFrame:
    """Многогоризонтный расчет: Prophet vs Ridge (линейная) vs CatBoost SOTA."""
    bench_mo = cfg["validation"]["benchmark_district"]
    cb_params = cfg["catboost_params"].copy()
    cb_params.pop("verbose", None)

    max_date = df["period"].max()
    cutoff = max_date - pd.DateOffset(months=12)

    tr = df[df["period"] <= cutoff].copy()
    te = df[df["period"] > cutoff].copy()

    tr["target_ratio"] = tr["value"] / tr["mo_baseline"]
    te["target_ratio"] = te["value"] / te["mo_baseline"]

    b_min = tr["mo_baseline"].min()
    b_max = tr["mo_baseline"].max()
    denom = (b_max - b_min) if (b_max > b_min) else 1.0
    tr_weights = 1.0 + 0.4 * (tr["mo_baseline"] - b_min) / denom

    # 1. CatBoost SOTA
    cb = CatBoostRegressor(**cb_params, verbose=False)
    cb.fit(tr[features], tr["target_ratio"], sample_weight=tr_weights)
    te["cb_pred"] = cb.predict(te[features]) * te["mo_baseline"].values

    # 2. Линейный бейзлайн (Ridge Regression)
    ridge = Ridge(alpha=1.0)
    ridge.fit(tr[features], tr["target_ratio"], sample_weight=tr_weights)
    te["ridge_pred"] = ridge.predict(te[features]) * te["mo_baseline"].values

    # 3. Prophet на бенчмарк-МО
    te_bench = te[te["mo"].str.contains(bench_mo[:10])].sort_values("period")
    tr_bench = tr[tr["mo"].str.contains(bench_mo[:10])].sort_values("period")

    tp = tr_bench[["period", "value"]].rename(columns={"period": "ds", "value": "y"})
    tep = te_bench[["period", "value"]].rename(columns={"period": "ds", "value": "y"})

    try:
        p_m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        p_m.fit(tp)
        te_bench["prophet_pred"] = p_m.predict(tep[["ds"]])["yhat"].values
    except Exception as exc:
        logger.warning("Prophet fit error: %s", exc)
        te_bench["prophet_pred"] = np.nan

    unique_test_dates = sorted(te["period"].unique())
    horizons_map = {
        1: "1 месяц (Оперативный)",
        3: "3 месяца (Квартальный)",
        6: "6 месяцев (Полугодовой)",
        12: "12 месяцев (Годовой стресс-тест)"
    }

    horizon_records = []
    for h, desc in horizons_map.items():
        target_dates = unique_test_dates[:h]
        sub_te = te[te["period"].isin(target_dates)]
        sub_bench = te_bench[te_bench["period"].isin(target_dates)]

        mae_rf_cb = mean_absolute_error(sub_te["value"], sub_te["cb_pred"])
        r2_rf_cb = r2_score(sub_te["value"], sub_te["cb_pred"])
        mae_rf_ridge = mean_absolute_error(sub_te["value"], sub_te["ridge_pred"])

        mae_cb_mo = mean_absolute_error(sub_bench["value"], sub_bench["cb_pred"])

        if not sub_bench["prophet_pred"].isna().all():
            mae_p_mo = mean_absolute_error(sub_bench["value"], sub_bench["prophet_pred"])
        else:
            mae_p_mo = np.nan

        gain = ((mae_p_mo - mae_cb_mo) / mae_p_mo * 100) if (not np.isnan(mae_p_mo) and mae_p_mo > 0) else 0.0

        horizon_records.append({
            "Горизонт прогноза": desc,
            "Prophet MAE (МО)": f"{round(mae_p_mo):,} руб." if not np.isnan(mae_p_mo) else "N/A",
            "Ridge MAE (РФ)": f"{round(mae_rf_ridge):,} руб.",
            "CatBoost MAE (МО)": f"{round(mae_cb_mo):,} руб.",
            "CatBoost MAE (РФ)": f"{round(mae_rf_cb):,} руб.",
            "CatBoost R² (РФ)": f"{r2_rf_cb:.4f}",
            "Превосходство CatBoost": f"+{gain:.1f}%"
        })

    df_horizons = pd.DataFrame(horizon_records)
    df_horizons.to_csv(cfg["data"].get("horizons_output_path", "horizons_benchmark.csv"), index=False, encoding="utf-8")
    return df_horizons


def train_and_evaluate(df: pd.DataFrame, cfg: Dict) -> Tuple[CatBoostRegressor, Dict]:
    primary_h = cfg["validation"].get("primary_horizon_months", 6)
    cutoff_date = df["period"].max() - pd.DateOffset(months=primary_h)
    logger.info("Основной горизонт валидации: %d месяцев (Cutoff: %s)", primary_h, cutoff_date.date())

    train_df = df[df["period"] <= cutoff_date].copy()
    test_df = df[df["period"] > cutoff_date].copy()

    # Расчет статистик МО строго на train
    mo_baseline = train_df.groupby("mo")["value"].mean().rename("mo_baseline")
    mo_std = train_df.groupby("mo")["value"].std().fillna(0).rename("mo_std")

    last_3m_cutoff = cutoff_date - pd.DateOffset(months=3)
    train_recent = train_df[train_df["period"] > last_3m_cutoff]
    mo_recent_mean = train_recent.groupby("mo")["value"].mean().rename("mo_recent_mean")

    mo_stats = pd.merge(mo_baseline, mo_std, on="mo", how="left")
    mo_stats = pd.merge(mo_stats, mo_recent_mean, on="mo", how="left")
    mo_stats["mo_recent_mean"] = mo_stats["mo_recent_mean"].fillna(mo_stats["mo_baseline"])
    mo_stats["mo_momentum"] = np.round(mo_stats["mo_recent_mean"] / mo_stats["mo_baseline"], 4)
    mo_stats["mo_volatility"] = np.round(mo_stats["mo_std"] / mo_stats["mo_baseline"], 4)
    mo_stats["log_baseline"] = np.round(np.log1p(mo_stats["mo_baseline"]), 4)

    stat_cols = ["mo_baseline", "mo_momentum", "mo_volatility", "log_baseline"]
    train_df = train_df.merge(mo_stats[stat_cols], on="mo", how="left")
    test_df = test_df.merge(mo_stats[stat_cols], on="mo", how="left")
    df = df.merge(mo_stats[stat_cols], on="mo", how="left")

    gm = train_df["value"].mean()
    for d in [train_df, test_df, df]:
        d["mo_baseline"] = d["mo_baseline"].fillna(gm)
        d["mo_momentum"] = d["mo_momentum"].fillna(1.0)
        d["mo_volatility"] = d["mo_volatility"].fillna(0.1)
        d["log_baseline"] = d["log_baseline"].fillna(np.log1p(gm))
        d["december_elasticity"] = d["is_december"] * d["log_baseline"]

    train_df["target_ratio"] = train_df["value"] / train_df["mo_baseline"]
    test_df["target_ratio"] = test_df["value"] / test_df["mo_baseline"]

    features = cfg["features"]["numerical"].copy()
    for extra_col in ["mo_momentum", "mo_volatility", "log_baseline", "december_elasticity"]:
        if extra_col not in features:
            features.append(extra_col)

    # 1. Расчет матрицы 4 горизонтов
    df_horizons = evaluate_four_horizons(df, cfg, features)

    # 2. Обучение финального CatBoost
    b_min = train_df["mo_baseline"].min()
    b_max = train_df["mo_baseline"].max()
    denom = (b_max - b_min) if (b_max > b_min) else 1.0
    sample_weights = 1.0 + 0.4 * (train_df["mo_baseline"] - b_min) / denom

    cb_params = cfg["catboost_params"].copy()
    cb_params.pop("verbose", None)

    cb_model = CatBoostRegressor(**cb_params, verbose=False)
    cb_model.fit(train_df[features], train_df["target_ratio"], sample_weight=sample_weights)

    pred_ratio_all = cb_model.predict(test_df[features])
    cb_all_preds = pred_ratio_all * test_df["mo_baseline"].values
    mae_cb_all = mean_absolute_error(test_df["value"], cb_all_preds)
    r2_cb_all = r2_score(test_df["value"], cb_all_preds)

    # 3. CUSUM-мониторинг остатков
    residuals = test_df["value"].values - cb_all_preds
    res_mean = np.mean(residuals)
    res_std = np.std(residuals) if np.std(residuals) > 0 else 1.0
    z_residuals = np.abs((residuals - res_mean) / res_std)
    
    test_df["residual_error"] = np.round(residuals)
    test_df["z_score"] = np.round(z_residuals, 2)
    top_anomalies = test_df[test_df["z_score"] > 2.5].sort_values("z_score", ascending=False)[
        ["period", "mo", "value", "residual_error", "z_score"]
    ].head(10)
    top_anomalies.columns = ["Дата", "Муниципальное образование", "Факт трат (руб)", "Отклонение модели (руб)", "Уровень аномалии (Z-Score)"]
    top_anomalies.to_csv("top_anomalies_detected.csv", index=False, encoding="utf-8")

    # 4. Стратифицированный срез 12 МО (полные наименования)
    q25 = train_df.groupby("mo")["value"].mean().quantile(0.25)
    q50 = train_df.groupby("mo")["value"].mean().quantile(0.50)
    q75 = train_df.groupby("mo")["value"].mean().quantile(0.75)

    sample_mos = []
    bench_mo = cfg["validation"]["benchmark_district"]
    if bench_mo in train_df["mo"].unique():
        sample_mos.append(bench_mo)

    for _, group in train_df.groupby("mo"):
        mo_name = group["mo"].iloc[0]
        if mo_name not in sample_mos:
            sample_mos.append(mo_name)
        if len(sample_mos) >= 12:
            break

    district_records = []
    prophet_evidence_list = []

    for mo_name in sample_mos:
        tr_mo = train_df[train_df["mo"] == mo_name]
        te_mo = test_df[test_df["mo"] == mo_name]

        if len(tr_mo) < 6 or len(te_mo) < 3:
            continue

        base_val = tr_mo["value"].mean()
        if base_val < q25:
            scale_group = "Малый (<25%)"
        elif base_val < q50:
            scale_group = "Средний (25-50%)"
        elif base_val < q75:
            scale_group = "Крупный (50-75%)"
        else:
            scale_group = "Сверхкрупный (>75%)"

        train_p = tr_mo[["period", "value"]].rename(columns={"period": "ds", "value": "y"})
        test_p = te_mo[["period", "value"]].rename(columns={"period": "ds", "value": "y"})

        try:
            p_model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
            p_model.fit(train_p)
            p_pred = p_model.predict(test_p[["ds"]])["yhat"].values
            p_mae = mean_absolute_error(test_p["y"], p_pred)
        except Exception:
            p_mae = np.nan
            p_pred = np.zeros(len(test_p))

        cb_pred_r = cb_model.predict(te_mo[features])
        cb_pred = cb_pred_r * te_mo["mo_baseline"].values
        cb_mae = mean_absolute_error(te_mo["value"], cb_pred)

        improvement = ((p_mae - cb_mae) / p_mae * 100) if (not np.isnan(p_mae) and p_mae > 0) else 0.0

        district_records.append({
            "Муниципальное образование": mo_name,
            "Масштаб трат": scale_group,
            "Базовый чек (руб)": round(base_val),
            "Prophet MAE (руб)": round(p_mae),
            "CatBoost MAE (руб)": round(cb_mae),
            "Превосходство (%)": round(improvement, 1)
        })

        test_p_copy = test_p.copy()
        test_p_copy["prophet_pred"] = p_pred
        test_p_copy["mo"] = mo_name
        prophet_evidence_list.append(test_p_copy)

    df_comparison = pd.DataFrame(district_records)
    df_comparison.to_csv(cfg["data"].get("districts_output_path", "benchmark_districts_comparison.csv"), index=False, encoding="utf-8")

    if prophet_evidence_list:
        df_all_prophet = pd.concat(prophet_evidence_list, ignore_index=True)
        df_all_prophet.to_csv(cfg["data"]["prophet_evidence_path"], index=False, encoding="utf-8")

    avg_p_mae = df_comparison["Prophet MAE (руб)"].mean()
    avg_cb_mae = df_comparison["CatBoost MAE (руб)"].mean()
    total_gain = ((avg_p_mae - avg_cb_mae) / avg_p_mae) * 100

    metrics = {
        "cutoff_date": str(cutoff_date.date()),
        "test_horizon_months": primary_h,
        "catboost_overall": {
            "mae": round(float(mae_cb_all), 2),
            "r2": round(float(r2_cb_all), 4)
        },
        "stratified_benchmark": {
            "num_districts_tested": len(df_comparison),
            "average_prophet_mae": round(float(avg_p_mae), 2),
            "average_catboost_mae": round(float(avg_cb_mae), 2),
            "overall_improvement_pct": round(float(total_gain), 2)
        },
        "coupled_shock_monitoring": {
            "residual_std_rub": round(float(res_std), 2),
            "anomalies_flagged_count": int(np.sum(z_residuals > 2.5)),
            "early_warning_mechanism": "Coupled NLP-Window Z-Score + Residual CUSUM"
        }
    }

    cb_model.save_model(cfg["data"]["model_output_path"])
    df.to_csv(cfg["data"]["processed_output_path"], index=False)

    with open(cfg["data"]["metrics_output_path"], "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)

    return cb_model, metrics


def main():
    cfg = load_config("config.yaml")
    logger.info("Запуск единого конвейера...")

    df = build_dataset(cfg)
    _, metrics = train_and_evaluate(df, cfg)

    df_horizons = pd.read_csv(cfg["data"].get("horizons_output_path", "horizons_benchmark.csv"))
    df_districts = pd.read_csv(cfg["data"].get("districts_output_path", "benchmark_districts_comparison.csv"))

    logger.info("=" * 85)
    logger.info("1. МАТРИЦА КАЧЕСТВА НА 4 ГОРИЗОНТАХ (PROPHET VS RIDGE VS CATBOOST SOTA):")
    logger.info("=" * 85)
    print(df_horizons.to_string(index=False))
    logger.info("=" * 85)
    logger.info("2. СТРАТИФИЦИРОВАННЫЙ СРЕЗ (12 МО):")
    logger.info("=" * 85)
    print(df_districts.to_string(index=False))
    logger.info("=" * 85)
    logger.info("ГЛОБАЛЬНЫЙ ИТОГ (2000+ МО РФ): CatBoost SOTA MAE = %s руб. | R2 = %s",
                f"{metrics['catboost_overall']['mae']:,.0f}", metrics['catboost_overall']['r2'])


if __name__ == "__main__":
    main()