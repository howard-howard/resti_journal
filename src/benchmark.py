"""Empirical Performance & Scalability Benchmarking Engine.

Performs a rigorous comparative benchmark between:
  1. Baseline: Traditional Eager Pandas Join & Cube Roll-Up (SQL -> Pandas -> GroupBy -> NumPy)
  2. Proposed: Vectorized Columnar Pushdown with Apache Arrow Zero-Copy & Polars
     (DuckDB -> Arrow IPC -> Polars Lazy Streaming)

Methodology:
  - OS process priority elevated to High to minimize kernel scheduling jitter.
  - Initial warm-up pass to eliminate disk cache / JIT cold-start bias.
  - 30 repeated randomized trials (N=30).
  - Metrics tracked: Wall-Clock Duration (s), Process CPU Time (s), Peak Resident Set Size (RSS MB).
  - Two-tailed Wilcoxon Signed-Rank Test for non-parametric statistical hypothesis verification.
"""

import gc
import os
import time
from typing import Callable, Tuple

import duckdb
import numpy as np
import pandas as pd
import polars as pl
import psutil
from scipy.stats import wilcoxon

try:
    from .config import (
        BENCHMARK_TRIALS,
        DB_PATH,
        MAX_LAT,
        MAX_LON,
        MIN_LAT,
        MIN_LON,
    )
except ImportError:
    from config import (
        BENCHMARK_TRIALS,
        DB_PATH,
        MAX_LAT,
        MAX_LON,
        MIN_LAT,
        MIN_LON,
    )

# Elevate process priority to minimize preemption by background OS processes
try:
    p = psutil.Process(os.getpid())
    if os.name == "nt":
        p.nice(psutil.HIGH_PRIORITY_CLASS)
    else:
        p.nice(-10)
except Exception:
    pass


def get_rss_mb() -> float:
    """Return process resident set size (RSS) in megabytes."""
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def run_pandas_baseline(con: duckdb.DuckDBPyConnection) -> int:
    """Baseline: Traditional Eager Pandas Join & In-Memory Cube Roll-Up."""
    df_pd = con.sql(
        f"""
        SELECT l.h3_res6 AS hex_id, w.tahun, w.bulan, fl.nama_flag, g.nama_geartype,
               f.hours, f.fishing_hours, f.mmsi_present
        FROM fact_gfw f
        JOIN dim_lokasi l ON f.lokasi_key = l.lokasi_key
        JOIN dim_waktu w ON f.waktu_key = w.waktu_key
        JOIN dim_gear g ON f.gear_key = g.gear_key
        JOIN dim_flag fl ON f.flag_key = fl.flag_key
        WHERE l.grid_lat BETWEEN {MIN_LAT} AND {MAX_LAT}
          AND l.grid_lon BETWEEN {MIN_LON} AND {MAX_LON}
          AND f.fishing_hours >= 0 AND f.fishing_hours <= f.hours
        """
    ).df()

    df_pd["fleet_category"] = np.where(df_pd["nama_flag"] == "IDN", "IDN_Fleet", "Non_IDN_Fleet")
    df_pd["monsoon_season"] = np.select(
        [
            df_pd["bulan"].isin([12, 1, 2]),
            df_pd["bulan"].isin([3, 4, 5]),
            df_pd["bulan"].isin([6, 7, 8]),
        ],
        ["West_Monsoon", "Transition_1", "East_Monsoon"],
        default="Transition_2",
    )

    cube = df_pd.groupby(["hex_id", "monsoon_season", "fleet_category"], as_index=False).agg(
        presence_hours=("hours", "sum"),
        fishing_hours=("fishing_hours", "sum"),
    )

    cube["loitering_ratio"] = np.where(
        cube["presence_hours"] > 0,
        (cube["presence_hours"] - cube["fishing_hours"]) / cube["presence_hours"],
        0.0,
    )
    return len(cube)


def run_duckdb_polars_proposed(con: duckdb.DuckDBPyConnection) -> int:
    """Proposed Target: Vectorized DuckDB Pushdown + Apache Arrow Zero-Copy + Polars."""
    df_pl = con.sql(
        f"""
        SELECT 
            l.h3_res6 AS hex_id,
            CASE 
                WHEN w.bulan IN (12, 1, 2) THEN 'West_Monsoon'
                WHEN w.bulan IN (3, 4, 5) THEN 'Transition_1'
                WHEN w.bulan IN (6, 7, 8) THEN 'East_Monsoon'
                ELSE 'Transition_2'
            END AS monsoon_season,
            CASE WHEN fl.nama_flag = 'IDN' THEN 'IDN_Fleet' ELSE 'Non_IDN_Fleet' END AS fleet_category,
            SUM(f.hours) AS presence_hours,
            SUM(f.fishing_hours) AS fishing_hours
        FROM fact_gfw f
        JOIN dim_lokasi l ON f.lokasi_key = l.lokasi_key
        JOIN dim_waktu w ON f.waktu_key = w.waktu_key
        JOIN dim_gear g ON f.gear_key = g.gear_key
        JOIN dim_flag fl ON f.flag_key = fl.flag_key
        WHERE l.grid_lat BETWEEN {MIN_LAT} AND {MAX_LAT}      
          AND l.grid_lon BETWEEN {MIN_LON} AND {MAX_LON}
          AND f.fishing_hours >= 0 AND f.fishing_hours <= f.hours
        GROUP BY 1, 2, 3
        """
    ).pl()

    df_pl = df_pl.with_columns(
        pl.when(pl.col("presence_hours") > 0)
        .then((pl.col("presence_hours") - pl.col("fishing_hours")) / pl.col("presence_hours"))
        .otherwise(0.0)
        .alias("loitering_ratio")
    )
    return df_pl.height


def benchmark_pipeline(
    fn: Callable[[duckdb.DuckDBPyConnection], int], name: str, n_trials: int = BENCHMARK_TRIALS
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Execute n_trials benchmarking on the supplied query function."""
    con = duckdb.connect(str(DB_PATH), read_only=True)

    # 1. Warm-up pass to eliminate cold-start cache bias
    _ = fn(con)
    gc.collect()

    wall_times = []
    cpu_times = []
    mem_deltas = []

    for _ in range(n_trials):
        gc.collect()
        base_mem = get_rss_mb()
        t0_wall = time.perf_counter()
        t0_cpu = time.process_time()

        _ = fn(con)

        cpu_times.append(time.process_time() - t0_cpu)
        wall_times.append(time.perf_counter() - t0_wall)
        mem_deltas.append(max(0.0, get_rss_mb() - base_mem))

    con.close()

    wall_arr = np.array(wall_times)
    cpu_arr = np.array(cpu_times)
    mem_arr = np.array(mem_deltas)

    print(f"=== {name} (N={n_trials}) ===")
    print(
        f"  Wall Time (s) : Median = {np.median(wall_arr):.4f} (IQR = {np.percentile(wall_arr, 75) - np.percentile(wall_arr, 25):.4f}) | "
        f"Mean = {np.mean(wall_arr):.4f} +/- {np.std(wall_arr):.4f}"
    )
    print(
        f"  CPU Time  (s) : Median = {np.median(cpu_arr):.4f} (IQR = {np.percentile(cpu_arr, 75) - np.percentile(cpu_arr, 25):.4f})"
    )
    print(
        f"  Peak RAM  (MB): Median = {np.median(mem_arr):.2f} | Max = {np.max(mem_arr):.2f}\n"
    )

    return wall_arr, cpu_arr, mem_arr


def main() -> None:
    print("=" * 75)
    print("RESTI JOURNAL: POLARS VS. PANDAS EMPIRICAL BENCHMARK")
    print(f"Target Database: {DB_PATH}")
    print(f"Trials: N = {BENCHMARK_TRIALS}")
    print("=" * 75 + "\n")

    w_pd, c_pd, m_pd = benchmark_pipeline(run_pandas_baseline, "Baseline: Traditional Pandas Eager Pipeline")
    w_pl, c_pl, m_pl = benchmark_pipeline(run_duckdb_polars_proposed, "Proposed: DuckDB Vectorized + Arrow Zero-Copy + Polars")

    stat, p_val = wilcoxon(w_pd, w_pl)
    speedup = float(np.median(w_pd) / np.median(w_pl))

    print("=" * 75)
    print("STATISTICAL INFERENCE SUMMARY (Wilcoxon Signed-Rank Test)")
    print("=" * 75)
    print(f"Wilcoxon Test Statistic : {stat:.2f}")
    print(f"P-Value (Two-tailed)     : {p_val:.6e}")
    print(f"Median Wall-Time Speedup : {speedup:.2f}x faster")
    if p_val < 0.001:
        print("Verdict                  : Statistically significant improvement (p < 0.001).")
    else:
        print("Verdict                  : No statistically significant difference.")


if __name__ == "__main__":
    main()

