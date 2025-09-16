#!/usr/bin/env python3
"""
Real-Time-Disaster-Impact-Analytics-Platform (Synthetic Data Demo)
------------------------------------------------------------------
This standalone script simulates a streaming disaster feed and computes
real-time impact analytics on a synthetic asset inventory (>100 points).

Features
- Generates > 1,000 synthetic assets (lat/lon, population, vulnerability).
- Simulates a stream of hazard events (e.g., flood/wind) with spatial footprints.
- Computes per-asset impact in (near) real time with decay by distance.
- Rolling analytics (per-minute aggregate, anomaly detection via z-score).
- Saves CSV artifacts and a PNG map of the latest impacts.

USAGE (no dependencies beyond numpy/pandas/matplotlib):
    python real_time_disaster_platform.py

Outputs created in ./outputs/ :
    - assets.csv
    - hazard_events.csv
    - impacts_stream.csv                (long format per event-asset impact)
    - impact_summary_by_minute.csv
    - latest_impact_map.png

This demo uses a Niger Delta–like bounding box for geospatial realism.
Author: ChatGPT
"""

from __future__ import annotations
import os
import math
import time
import uuid
import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------- Configuration -------------------------------

RNG_SEED             = 42
ASSET_COUNT          = 1500        # > 100 points (synthetic assets)
EVENT_COUNT          = 240         # ~4 hours at 1/min if you like to imagine
ASSET_BOUNDS         = {           # Niger Delta-ish bbox (approx)
    "lat_min": 4.3, "lat_max": 6.2,
    "lon_min": 5.0, "lon_max": 7.5
}
HAZARD_TYPES         = ["flood", "wind"]
EVENT_RADIUS_RANGE_KM = (10, 80)   # spatial footprint radius
EVENT_INTENSITY_RANGE = (0.4, 1.0) # base intensity multiplier
CRITICAL_PROB        = 0.1         # % of assets as critical infrastructure
ROLLING_WINDOW_MIN    = 15         # analytics window for anomaly detection

OUTPUT_DIR            = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

np.random.seed(RNG_SEED)
random.seed(RNG_SEED)


# ------------------------------- Data Classes --------------------------------

@dataclass
class HazardEvent:
    event_id: str
    timestamp: pd.Timestamp
    hazard_type: str
    center_lat: float
    center_lon: float
    radius_km: float
    base_intensity: float


# ------------------------------- Generators ----------------------------------

def generate_assets(n: int) -> pd.DataFrame:
    """Generate synthetic assets within a bounding box."""
    lat = np.random.uniform(ASSET_BOUNDS["lat_min"], ASSET_BOUNDS["lat_max"], n)
    lon = np.random.uniform(ASSET_BOUNDS["lon_min"], ASSET_BOUNDS["lon_max"], n)

    # Population and vulnerability
    population = np.random.lognormal(mean=6.5, sigma=0.6, size=n).astype(int)  # ~exp(6.5) ~ 665
    vulnerability = np.clip(np.random.normal(0.5, 0.15, n), 0.05, 0.95)

    # Infrastructure criticality
    critical = np.random.rand(n) < CRITICAL_PROB

    assets = pd.DataFrame({
        "asset_id": [f"A{i:05d}" for i in range(n)],
        "lat": lat,
        "lon": lon,
        "population": population,
        "vulnerability": vulnerability,
        "is_critical": critical
    })
    return assets


def simulate_event(t0: pd.Timestamp, minute_offset: int) -> HazardEvent:
    """Create a single hazard event at t0 + minute_offset minutes."""
    hazard_type = random.choice(HAZARD_TYPES)

    # Centers biased toward clusters (hotspots) for realism
    hotspot_centers = [
        (5.5, 6.7),  # near Port Harcourt-ish
        (5.3, 5.5),  # coastal area
        (6.0, 7.2)   # inland
    ]
    if random.random() < 0.7:
        base_lat, base_lon = random.choice(hotspot_centers)
        jitter = np.random.normal(0, 0.15, 2)
        center_lat = float(np.clip(base_lat + jitter[0], ASSET_BOUNDS["lat_min"], ASSET_BOUNDS["lat_max"]))
        center_lon = float(np.clip(base_lon + jitter[1], ASSET_BOUNDS["lon_min"], ASSET_BOUNDS["lon_max"]))
    else:
        center_lat = float(np.random.uniform(ASSET_BOUNDS["lat_min"], ASSET_BOUNDS["lat_max"]))
        center_lon = float(np.random.uniform(ASSET_BOUNDS["lon_min"], ASSET_BOUNDS["lon_max"]))

    radius_km = float(np.random.uniform(*EVENT_RADIUS_RANGE_KM))
    base_intensity = float(np.random.uniform(*EVENT_INTENSITY_RANGE))

    return HazardEvent(
        event_id=str(uuid.uuid4())[:8],
        timestamp=t0 + pd.Timedelta(minutes=minute_offset),
        hazard_type=hazard_type,
        center_lat=center_lat,
        center_lon=center_lon,
        radius_km=radius_km,
        base_intensity=base_intensity
    )


def haversine_km(lat1, lon1, lat2, lon2):
    """Calculate great-circle distance between two points on Earth (km)."""
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2.0)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def event_impact_on_assets(event: HazardEvent, assets: pd.DataFrame) -> pd.DataFrame:
    """Compute impact of a single event on all assets with radial decay."""
    # Vectorized distance calc
    d_km = haversine_km(event.center_lat, event.center_lon, assets["lat"].values, assets["lon"].values)

    # Only assets within ~3 * radius feel some impact (soft cutoff)
    influence_mask = d_km <= (event.radius_km * 3)
    affected = assets.loc[influence_mask].copy()
    if affected.empty:
        return pd.DataFrame(columns=[
            "event_id","asset_id","timestamp","hazard_type","distance_km",
            "hazard_intensity","impact_score","impact_severity"
        ])

    # Intensity decays with distance (Gaussian-like), clipping small values
    intensity = event.base_intensity * np.exp(-(d_km[influence_mask] / event.radius_km)**2)
    intensity = np.clip(intensity, 0, None)

    # Impact combines intensity, population exposure, and vulnerability
    exposure = affected["population"].values ** 0.5   # diminishing returns on population
    vulnerability = affected["vulnerability"].values
    critical_boost = np.where(affected["is_critical"].values, 1.25, 1.0)

    impact = intensity * exposure * vulnerability * critical_boost

    # Normalize impact to a 0..100 scale for interpretability
    if impact.max() > 0:
        impact_norm = 100 * (impact / impact.max())
    else:
        impact_norm = impact

    # Severity bands
    bins = [-np.inf, 5, 20, 40, 70, np.inf]
    labels = ["Minimal", "Minor", "Moderate", "Severe", "Extreme"]
    severity = pd.cut(impact_norm, bins=bins, labels=labels)

    out = pd.DataFrame({
        "event_id": event.event_id,
        "asset_id": affected["asset_id"].values,
        "timestamp": event.timestamp,
        "hazard_type": event.hazard_type,
        "distance_km": np.round(d_km[influence_mask], 2),
        "hazard_intensity": np.round(intensity, 4),
        "impact_score": np.round(impact_norm, 2),
        "impact_severity": severity.astype(str)
    })
    return out


# ------------------------------ Analytics Utils ------------------------------

def zscore_anomalies(series: pd.Series, window: int = ROLLING_WINDOW_MIN, z_thresh: float = 2.5) -> pd.Series:
    """Return a boolean Series where values are anomalous by rolling z-score."""
    roll_mean = series.rolling(window=window, min_periods=max(3, window//3)).mean()
    roll_std  = series.rolling(window=window, min_periods=max(3, window//3)).std(ddof=0)
    z = (series - roll_mean) / (roll_std.replace(0, np.nan))
    return (z.abs() >= z_thresh).fillna(False)


def summarize_per_minute(impacts: pd.DataFrame) -> pd.DataFrame:
    """Aggregate impacts per minute and compute anomaly flags."""
    g = impacts.groupby(pd.Grouper(key="timestamp", freq="1min"))
    summary = g.agg(
        events=("event_id", "nunique"),
        affected_assets=("asset_id", "nunique"),
        mean_impact=("impact_score", "mean"),
        p90_impact=("impact_score", lambda x: np.nanpercentile(x, 90)),
        extreme_hits=("impact_severity", lambda s: (s == "Extreme").sum()),
        severe_hits=("impact_severity", lambda s: (s == "Severe").sum()),
    ).reset_index()

    # Fill NA for early windows
    summary[["mean_impact","p90_impact"]] = summary[["mean_impact","p90_impact"]].fillna(0)

    # Anomaly flags
    summary["anomaly_affected_assets"] = zscore_anomalies(summary["affected_assets"])
    summary["anomaly_extreme"] = zscore_anomalies(summary["extreme_hits"])
    summary["anomaly_p90"] = zscore_anomalies(summary["p90_impact"])

    return summary


# --------------------------------- Plotting ----------------------------------

def plot_latest_map(assets: pd.DataFrame, impacts: pd.DataFrame, outpath: str):
    """Scatter plot of latest-minute impacts over asset locations."""
    if impacts.empty:
        return

    latest_minute = impacts["timestamp"].max().floor("min")
    latest = impacts[impacts["timestamp"].dt.floor("min") == latest_minute]

    # Join for coordinates
    latest = latest.merge(assets[["asset_id","lat","lon"]], on="asset_id", how="left")

    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(latest["lon"], latest["lat"],
                    s=10 + latest["impact_score"] * 0.8,
                    c=latest["impact_score"],
                    alpha=0.8)
    plt.colorbar(sc, ax=ax, label="Impact Score (0-100)")

    ax.set_title(f"Latest Impacts @ {latest_minute} (n={len(latest)})")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(ASSET_BOUNDS["lon_min"]-0.1, ASSET_BOUNDS["lon_max"]+0.1)
    ax.set_ylim(ASSET_BOUNDS["lat_min"]-0.1, ASSET_BOUNDS["lat_max"]+0.1)
    ax.grid(True, linestyle="--", alpha=0.3)

    # Add a rough bbox frame label
    ax.text(0.02, 0.98, "Niger Delta (approx bbox)", transform=ax.transAxes,
            va="top", ha="left", fontsize=9, alpha=0.7)

    plt.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


# --------------------------------- Pipeline ----------------------------------

def main():
    # 1) Assets
    assets = generate_assets(ASSET_COUNT)
    assets_path = os.path.join(OUTPUT_DIR, "assets.csv")
    assets.to_csv(assets_path, index=False)

    # 2) Generate event stream
    start_time = pd.Timestamp.utcnow().floor("min")
    events: List[HazardEvent] = [simulate_event(start_time, i) for i in range(EVENT_COUNT)]
    events_df = pd.DataFrame([e.__dict__ for e in events])
    events_path = os.path.join(OUTPUT_DIR, "hazard_events.csv")
    events_df.to_csv(events_path, index=False)

    # 3) Stream processing (no sleeps in demo; iterate quickly)
    impact_rows = []
    for ev in events:
        impacts_ev = event_impact_on_assets(ev, assets)
        if not impacts_ev.empty:
            impact_rows.append(impacts_ev)

    if impact_rows:
        impacts = pd.concat(impact_rows, ignore_index=True)
    else:
        impacts = pd.DataFrame(columns=[
            "event_id","asset_id","timestamp","hazard_type","distance_km",
            "hazard_intensity","impact_score","impact_severity"
        ])

    impacts["timestamp"] = pd.to_datetime(impacts["timestamp"])
    impacts_path = os.path.join(OUTPUT_DIR, "impacts_stream.csv")
    impacts.to_csv(impacts_path, index=False)

    # 4) Minute-level analytics + anomaly detection
    summary = summarize_per_minute(impacts)
    summary_path = os.path.join(OUTPUT_DIR, "impact_summary_by_minute.csv")
    summary.to_csv(summary_path, index=False)

    # 5) Map of latest-minute impacts
    map_path = os.path.join(OUTPUT_DIR, "latest_impact_map.png")
    plot_latest_map(assets, impacts, map_path)

    # 6) Console recap
    print("Real-Time-Disaster-Impact-Analytics-Platform (Synthetic Demo)")
    print(f"Assets:          {len(assets):>6}  -> {assets_path}")
    print(f"Hazard events:   {len(events):>6}  -> {events_path}")
    print(f"Impact records:  {len(impacts):>6}  -> {impacts_path}")
    print(f"Minute summary:  {len(summary):>6}  -> {summary_path}")
    print(f"Latest map PNG:            -> {map_path}")

    # 7) Simple alert snapshot (top 5 minutes by p90 impact)
    top5 = summary.sort_values("p90_impact", ascending=False).head(5)
    print("\nTop 5 minutes by p90 impact:")
    print(top5[["timestamp","events","affected_assets","p90_impact","extreme_hits","anomaly_p90"]].to_string(index=False))

if __name__ == "__main__":
    main()
