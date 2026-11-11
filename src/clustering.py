"""Unsupervised Spatio-Temporal HDBSCAN Clustering of Fishing Fleet Dynamics.

Implements density-based clustering over Uber H3 discrete global grid cells
across Indonesia's Fisheries Management Area 711 (WPPNRI 711). Evaluates seasonal
segregation between domestic Indonesian fleets (IDN) and transboundary fleets (Non-IDN).

Features:
  1. Spatio-temporal cube aggregation via DuckDB and Polars.
  2. Loitering ratio calculation: (presence_hours - fishing_hours) / presence_hours.
  3. Extraction of Uber H3 resolution 6 cell centroid coordinates.
  4. Standardized 6D kinematic-geopolitical feature space:
       [centroid_lat, centroid_lon, fishing_IDN, fishing_Non_IDN, loiter_IDN, loiter_Non_IDN]
  5. HDBSCAN clustering with GLOSH outlier/anomaly score derivation.
  6. Cluster validation via Silhouette Coefficient and DBCV (Density-Based Clustering Validation).
"""

import time
from typing import List, Tuple

import duckdb
import h3
import hdbscan
import numpy as np
import polars as pl
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

try:
    from .config import (
        DB_PATH,
        FEATURE_COLUMNS,
        HDBSCAN_PARAMS,
        MAX_LAT,
        MAX_LON,
        MIN_LAT,
        MIN_LON,
        ZONES_CSV_PATH,
    )
except ImportError:
    from config import (
        DB_PATH,
        FEATURE_COLUMNS,
        HDBSCAN_PARAMS,
        MAX_LAT,
        MAX_LON,
        MIN_LAT,
        MIN_LON,
        ZONES_CSV_PATH,
    )


def load_seasonal_spatiotemporal_cube(db_path) -> pl.DataFrame:
    """Extract and aggregate H3 seasonal fleet cube from materialized Lakehouse database."""
    print(f"Connecting to DuckDB Lakehouse: {db_path}...")
    with duckdb.connect(str(db_path), read_only=True) as con:
        # Check available tables
        tables = [t[0] for t in con.execute("SHOW TABLES;").fetchall()]
        if "wpp711_stc_h3_flag" in tables:
            df_raw = con.sql("SELECT * FROM wpp711_stc_h3_flag").pl()
        elif "fact_gfw" in tables or "uts_fact_gfw" in tables:
            prefix = "uts_" if "uts_fact_gfw" in tables and "fact_gfw" not in tables else ""
            df_raw = con.sql(
                f"""
                SELECT
                    l.h3_res6 AS hex_id,
                    w.tahun,
                    w.bulan,
                    CASE
                        WHEN w.bulan IN (12, 1, 2) THEN 'West_Monsoon'
                        WHEN w.bulan IN (3, 4, 5) THEN 'Transition_1'
                        WHEN w.bulan IN (6, 7, 8) THEN 'East_Monsoon'
                        ELSE 'Transition_2'
                    END AS monsoon_season,
                    CASE
                        WHEN fl.nama_flag = 'IDN' THEN 'IDN_Fleet'
                        ELSE 'Non_IDN_Fleet'
                    END AS fleet_category,
                    g.nama_geartype,
                    SUM(f.hours) AS total_presence_hours,
                    SUM(f.fishing_hours) AS total_fishing_hours,
                    COUNT(*) AS vessel_days_count,
                    CASE
                        WHEN SUM(f.hours) > 0 THEN (SUM(f.hours) - SUM(f.fishing_hours)) / SUM(f.hours)
                        ELSE 0.0
                    END AS loitering_ratio
                FROM {prefix}fact_gfw f
                JOIN {prefix}dim_lokasi l ON f.lokasi_key = l.lokasi_key
                JOIN {prefix}dim_waktu w ON f.waktu_key = w.waktu_key
                JOIN {prefix}dim_gear g ON f.gear_key = g.gear_key
                JOIN {prefix}dim_flag fl ON f.flag_key = fl.flag_key
                WHERE l.grid_lat BETWEEN {MIN_LAT} AND {MAX_LAT}
                  AND l.grid_lon BETWEEN {MIN_LON} AND {MAX_LON}
                  AND f.hours > 0 AND f.fishing_hours >= 0 AND f.fishing_hours <= f.hours
                GROUP BY 1, 2, 3, 4, 5, 6
                """
            ).pl()
        else:
            raise KeyError(
                f"Expected table 'wpp711_stc_h3_flag' or star-schema tables not found in database. Available: {tables}"
            )

    print(f"Loaded {df_raw.height:,} raw seasonal cube records.")

    # Aggregate by hex_id, monsoon_season, and fleet_category
    df_seasonal = (
        df_raw.group_by(["hex_id", "monsoon_season", "fleet_category"])
        .agg(
            [
                pl.col("total_presence_hours").sum().alias("presence_hours"),
                pl.col("total_fishing_hours").sum().alias("fishing_hours"),
            ]
        )
        .with_columns(
            pl.when(pl.col("presence_hours") > 0)
            .then((pl.col("presence_hours") - pl.col("fishing_hours")) / pl.col("presence_hours"))
            .otherwise(0.0)
            .alias("loitering_ratio")
        )
    )

    # Pivot into separate columns per fleet category
    df_pivot = df_seasonal.pivot(
        index=["hex_id", "monsoon_season"],
        on="fleet_category",
        values=["presence_hours", "fishing_hours", "loitering_ratio"],
        aggregate_function="first",
    ).fill_null(0.0)

    # Guarantee all expected fleet columns exist even if a season has 0 records for a fleet
    expected_cols = [
        "presence_hours_IDN_Fleet",
        "presence_hours_Non_IDN_Fleet",
        "fishing_hours_IDN_Fleet",
        "fishing_hours_Non_IDN_Fleet",
        "loitering_ratio_IDN_Fleet",
        "loitering_ratio_Non_IDN_Fleet",
    ]
    for col_name in expected_cols:
        if col_name not in df_pivot.columns:
            df_pivot = df_pivot.with_columns(pl.lit(0.0).alias(col_name))

    # Derive H3 cell geographic centroids
    hex_ids = df_pivot["hex_id"].to_list()
    lat_list, lon_list = zip(*[h3.cell_to_latlng(h) for h in hex_ids])

    df_pivot = df_pivot.with_columns(
        [
            pl.Series(name="centroid_lat", values=lat_list),
            pl.Series(name="centroid_lon", values=lon_list),
        ]
    )

    return df_pivot


def run_hdbscan_clustering(
    df_season: pl.DataFrame, season_name: str
) -> Tuple[pl.DataFrame, dict]:
    """Execute HDBSCAN on a single monsoon season slice using standardized 6D features."""
    X_raw = df_season.select(FEATURE_COLUMNS).to_numpy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=HDBSCAN_PARAMS["min_cluster_size"],
        min_samples=HDBSCAN_PARAMS["min_samples"],
        metric=HDBSCAN_PARAMS["metric"],
        gen_min_span_tree=HDBSCAN_PARAMS["gen_min_span_tree"],
    )
    cluster_labels = clusterer.fit_predict(X_scaled)
    outlier_scores = np.nan_to_num(clusterer.outlier_scores_, nan=0.0)

    # Validation metrics
    mask = cluster_labels != -1
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = int(np.sum(~mask))

    sil_score = (
        float(silhouette_score(X_scaled[mask], cluster_labels[mask]))
        if len(set(cluster_labels[mask])) > 1
        else 0.0
    )
    dbcv_score = float(clusterer.relative_validity_)

    metrics = {
        "season": season_name,
        "n_hexagons": df_season.height,
        "clusters_found": n_clusters,
        "noise_hexagons": n_noise,
        "silhouette_score": sil_score,
        "dbcv_score": dbcv_score,
    }

    df_season = df_season.with_columns(
        [
            pl.Series(name="geopolitical_zone_cluster", values=cluster_labels),
            pl.Series(name="anomaly_score", values=outlier_scores),
        ]
    )
    return df_season, metrics


def main() -> None:
    t0 = time.perf_counter()
    print("=" * 75)
    print("RESTI JOURNAL: HDBSCAN SPATIO-TEMPORAL FLEET SEGREGATION PIPELINE")
    print("=" * 75)

    df_pivot = load_seasonal_spatiotemporal_cube(DB_PATH)
    seasons = sorted(df_pivot["monsoon_season"].unique().to_list())

    all_clustered_dfs: List[pl.DataFrame] = []
    summary_metrics: List[dict] = []

    print("\nExecuting Seasonal Clustering & Validation:")
    for season in seasons:
        df_season = df_pivot.filter(pl.col("monsoon_season") == season)
        df_clustered, metrics = run_hdbscan_clustering(df_season, season)
        all_clustered_dfs.append(df_clustered)
        summary_metrics.append(metrics)

        print(
            f"  [{metrics['season']:<14}] Hexes: {metrics['n_hexagons']:>5,} | "
            f"Clusters: {metrics['clusters_found']:>2} | Noise: {metrics['noise_hexagons']:>4,} | "
            f"Silhouette: {metrics['silhouette_score']:>6.4f} | DBCV: {metrics['dbcv_score']:>6.4f}"
        )

    df_final = pl.concat(all_clustered_dfs)
    df_final.write_csv(ZONES_CSV_PATH)
    print(f"\n[OK] Clustered zones exported to: {ZONES_CSV_PATH} ({df_final.height:,} rows)")
    print(f"Elapsed Time: {time.perf_counter() - t0:.2f}s")


if __name__ == "__main__":
    main()

