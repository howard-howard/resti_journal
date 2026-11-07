"""Geospatial Cartography & Marine Spatial Planning (MSP) Visualizer.

Generates publication-ready 300 DPI figures illustrating seasonal fleet segregation
between domestic Indonesian fleets and transboundary foreign fleets in WPPNRI 711
(Natuna Sea & Karimata Strait) derived from HDBSCAN clustering.
"""

import time
from pathlib import Path

import contextily as ctx
import geopandas as gpd
import h3
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import Polygon

try:
    from .config import (
        FIGURES_DIR,
        MAX_LAT,
        MAX_LON,
        MIN_LAT,
        MIN_LON,
        ZONES_CSV_PATH,
    )
except ImportError:
    from config import (
        FIGURES_DIR,
        MAX_LAT,
        MAX_LON,
        MIN_LAT,
        MIN_LON,
        ZONES_CSV_PATH,
    )

TARGET_SEASONS = [
    ("West_Monsoon", "HDBSCAN Fleet Segregation (West Monsoon)", "hdbscan_wpp711_west_monsoon.png"),
    ("Transition_1", "HDBSCAN Fleet Segregation (First Transition)", "hdbscan_wpp711_transition_1.png"),
    ("East_Monsoon", "HDBSCAN Fleet Segregation (East Monsoon)", "hdbscan_wpp711_east_monsoon.png"),
    ("Transition_2", "HDBSCAN Fleet Segregation (Second Transition)", "hdbscan_wpp711_transition_2.png"),
]

COLOR_MAP = {
    "Domestic Archipelagic Zones (IDN Dominant)": "#1f77b4",  # Blue
    "Transboundary Corridors (Non-IDN Dominant)": "#d62728",  # Red
    "Kinematic Outliers (Cluster -1 Noise)": "#2b2b2b",       # Charcoal
}


def hex_to_polygon(hex_id: str) -> Polygon:
    """Convert Uber H3 index into Shapely Polygon (EPSG:4326 coords)."""
    boundaries = h3.cell_to_boundary(hex_id)
    coords = [(lon, lat) for lat, lon in boundaries]
    return Polygon(coords)


def render_seasonal_maps(csv_path: Path, output_dir: Path) -> None:
    """Load clustered zones, build GeoDataFrame, classify cluster dominance, and plot."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Clustering output file not found: {csv_path}")

    print(f"Loading clustered dataset: {csv_path}...")
    df = pd.read_csv(csv_path)

    print("Constructing spatial geometries from Uber H3 indexes...")
    df["geometry"] = df["hex_id"].apply(hex_to_polygon)
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

    output_dir.mkdir(parents=True, exist_ok=True)

    for season_key, plot_title, out_filename in TARGET_SEASONS:
        out_filepath = output_dir / out_filename
        gdf_season = gdf[gdf["monsoon_season"] == season_key].copy()
        if gdf_season.empty:
            print(f"[SKIP] No records found for season: {season_key}")
            continue

        # Classify cluster dominance based on total fishing effort hours
        cluster_dominance = {}
        for cluster_id in gdf_season["geopolitical_zone_cluster"].unique():
            if cluster_id == -1:
                cluster_dominance[cluster_id] = "Kinematic Outliers (Cluster -1 Noise)"
                continue

            c_sub = gdf_season[gdf_season["geopolitical_zone_cluster"] == cluster_id]
            total_idn = c_sub["fishing_hours_IDN_Fleet"].sum()
            total_non_idn = c_sub["fishing_hours_Non_IDN_Fleet"].sum()

            if total_idn >= total_non_idn:
                cluster_dominance[cluster_id] = "Domestic Archipelagic Zones (IDN Dominant)"
            else:
                cluster_dominance[cluster_id] = "Transboundary Corridors (Non-IDN Dominant)"

        gdf_season["fleet_dominance"] = gdf_season["geopolitical_zone_cluster"].map(cluster_dominance)

        # Plotting figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))

        for dominance_label, color in COLOR_MAP.items():
            subset = gdf_season[gdf_season["fleet_dominance"] == dominance_label]
            if not subset.empty:
                is_noise = dominance_label == "Kinematic Outliers (Cluster -1 Noise)"
                subset.plot(
                    ax=ax,
                    color=color,
                    edgecolor="white" if is_noise else "black",
                    linewidth=0.05 if is_noise else 0.12,
                    alpha=0.60 if is_noise else 0.85,
                )

        # Basemap integration
        try:
            ctx.add_basemap(ax, crs=gdf_season.crs.to_string(), source=ctx.providers.Esri.WorldGrayCanvas)
        except Exception:
            ax.set_facecolor("#edf2f7")

        legend_handles = [
            mpatches.Patch(
                facecolor=COLOR_MAP["Domestic Archipelagic Zones (IDN Dominant)"],
                edgecolor="black",
                alpha=0.85,
                label="Domestic Archipelagic Zones (IDN Dominant)",
            ),
            mpatches.Patch(
                facecolor=COLOR_MAP["Transboundary Corridors (Non-IDN Dominant)"],
                edgecolor="black",
                alpha=0.85,
                label="Transboundary Corridors (Non-IDN Dominant)",
            ),
            mpatches.Patch(
                facecolor=COLOR_MAP["Kinematic Outliers (Cluster -1 Noise)"],
                edgecolor="white",
                alpha=0.65,
                label="Kinematic Outliers (Cluster -1 Noise)",
            ),
        ]
        ax.legend(
            handles=legend_handles,
            loc="lower left",
            bbox_to_anchor=(0.01, 0.04),
            frameon=True,
            facecolor="white",
            fontsize=10,
        )

        ax.set_title(plot_title, fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Longitude (°E)", fontsize=10)
        ax.set_ylabel("Latitude (°N)", fontsize=10)
        ax.set_xlim([MIN_LON, MAX_LON])
        ax.set_ylim([MIN_LAT, MAX_LAT])
        ax.grid(True, linestyle="--", linewidth=0.35, alpha=0.55)

        plt.tight_layout()
        plt.savefig(out_filepath, dpi=300)
        plt.close(fig)
        print(f"  [MAP GENERATED] {out_filepath.name} (300 DPI)")


def main() -> None:
    t0 = time.perf_counter()
    print("=" * 75)
    print("RESTI JOURNAL: CARTOGRAPHY ENGINE (WPPNRI 711 FLEET SEGREGATION)")
    print("=" * 75)
    render_seasonal_maps(ZONES_CSV_PATH, FIGURES_DIR)
    print(f"\nAll cartographic figures generated in: {time.perf_counter() - t0:.2f}s")


if __name__ == "__main__":
    main()

