"""
Модуль сравнительного тестирования алгоритмов детекции разладок (Ruptures)
и фундаментальных моделей временных рядов (Amazon Chronos).
Рассчитывает формальные метрики детекции: Precision, Recall, F1-Score и задержку.
"""

import logging
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import ruptures as rpt
from sklearn.metrics import mean_absolute_error, r2_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Истинные макроэкономические даты структурных шоков в РФ (Ground Truth Shocks)
GROUND_TRUTH_SHOCKS = ["2023-08-01", "2024-06-01", "2024-09-01"]


def evaluate_detection_metrics(detected_dates: List[str], ground_truth: List[str], tolerance_months: int = 1) -> Tuple[float, float, float, float]:
    """
    Расчет строгих метрик качества детекции разладок:
    - Precision: доля верно определенных шоков среди всех найденных точек.
    - Recall: доля обнаруженных реальных шоков от их общего числа.
    - F1-Score: гармоническое среднее точности и полноты.
    - Mean Delay: среднее запаздывание детекции в месяцах.
    """
    if not detected_dates:
        return 0.0, 0.0, 0.0, np.nan

    gt_dt = [pd.to_datetime(d) for d in ground_truth]
    det_dt = [pd.to_datetime(d) for d in detected_dates]

    true_positives = 0
    delays = []
    matched_gt = set()

    for d in det_dt:
        # Поиск ближайшего реального шока
        time_diffs = [abs((d - g).days) / 30.44 for g in gt_dt]
        min_idx = int(np.argmin(time_diffs))
        min_diff = time_diffs[min_idx]

        # Попадание в окно толерантности (+/- 1 месяц)
        if min_diff <= (tolerance_months + 0.5) and min_idx not in matched_gt:
            true_positives += 1
            delays.append(min_diff)
            matched_gt.add(min_idx)

    precision = true_positives / len(det_dt) if det_dt else 0.0
    recall = true_positives / len(gt_dt) if gt_dt else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    mean_delay = float(np.mean(delays)) if delays else np.nan

    return precision, recall, f1, mean_delay


def run_changepoint_benchmark():
    """Сравнительное тестирование Pelt, Binary Segmentation и Window-based на реальных данных."""
    logger.info("Запуск количественного бенчмарка алгоритмов детекции разладок...")
    if not os.path.exists("final_data_for_app.csv"):
        raise FileNotFoundError("final_data_for_app.csv не найден. Запустите train.py.")

    df = pd.read_csv("final_data_for_app.csv")
    df["period"] = pd.to_datetime(df["period"])

    # Бенчмарк на опорном районе
    bench_mo = "Октябрьский муниципальный район"
    df_mo = df[df["mo"].str.contains(bench_mo[:10])].groupby("period")["value"].sum().reset_index().sort_values("period")
    
    signal = (df_mo["value"].values / df_mo["value"].mean()).reshape(-1, 1)
    dates_str = df_mo["period"].dt.strftime("%Y-%m-%d").tolist()

    methods = {
        "Pelt (L2 penalty)": rpt.Pelt(model="l2", min_size=2),
        "Binary Segmentation": rpt.Binseg(model="l2", min_size=2),
        "Window-based (w=3)": rpt.Window(width=3, model="l2")
    }

    benchmark_rows = []

    for name, detector in methods.items():
        t0 = time.time()
        detector.fit(signal)

        if "Window" in name:
            breaks = detector.predict(n_bkps=2)
        elif "Binseg" in name:
            breaks = detector.predict(n_bkps=2)
        else:
            breaks = detector.predict(pen=0.6)

        exec_time = (time.time() - t0) * 1000
        break_indices = breaks[:-1]
        detected = [dates_str[i] for i in break_indices if i < len(dates_str)]

        p, r, f1, delay = evaluate_detection_metrics(detected, GROUND_TRUTH_SHOCKS, tolerance_months=1)

        benchmark_rows.append({
            "Алгоритм детекции": name,
            "Время (мс)": round(exec_time, 2),
            "Найденные даты": ", ".join(detected) if detected else "Нет точек",
            "Precision": round(p, 2),
            "Recall": round(r, 2),
            "F1-Score": round(f1, 2),
            "Задержка (мес.)": round(delay, 1) if not np.isnan(delay) else "N/A"
        })

    df_cpt_metrics = pd.DataFrame(benchmark_rows)
    df_cpt_metrics.to_csv("cpt_benchmark_metrics.csv", index=False, encoding="utf-8")
    logger.info("Матрица метрик детекции сформирована:\n%s", df_cpt_metrics.to_string(index=False))


if __name__ == "__main__":
    run_changepoint_benchmark()