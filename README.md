Real-Time Disaster Impact Analytics Platform
📌 Overview

This project is a synthetic data demonstration of a real-time disaster impact analytics system. It simulates natural hazard events (e.g., floods and windstorms) and computes their spatiotemporal impacts on vulnerable assets within the Niger Delta region.

It integrates:

🛰️ Geospatial data modeling

🤖 Impact estimation using vulnerability, population, and criticality factors

📈 Rolling real-time analytics and anomaly detection

🗺️ Visualization of disaster impact hotspots

This framework serves as a foundation for building operational disaster monitoring systems for climate resilience, early warning, and risk mitigation.

⚙️ Features

Generate >1,000 synthetic asset points with attributes (location, population, vulnerability).

Simulate real-time disaster event streams (flood/wind).

Compute impact severity scores based on exposure and vulnerability.

Perform rolling-minute analytics and z-score anomaly detection.

Output:

assets.csv — synthetic assets database

hazard_events.csv — simulated disaster events

impacts_stream.csv — asset-level event impacts

impact_summary_by_minute.csv — aggregated rolling analytics

latest_impact_map.png — visualization of recent impacts

🧠 Methodology

Impact Score Formula

Impact = hazard_intensity × sqrt(population) × vulnerability × critical_boost


hazard_intensity: Decays with distance from event center (Gaussian-like)

critical_boost: +25% for critical infrastructure

Normalized to 0–100 and categorized into severity bands: Minimal, Minor, Moderate, Severe, Extreme

Anomaly Detection

Rolling 15-minute window

Z-score ≥ 2.5 triggers anomaly flags on:

Number of affected assets

Extreme events

90th percentile impact

📂 Project Structure
real_time_disaster_platform.py
outputs/
 ├─ assets.csv
 ├─ hazard_events.csv
 ├─ impacts_stream.csv
 ├─ impact_summary_by_minute.csv
 └─ latest_impact_map.png
README.md

🚀 Usage
1. Requirements

Python 3.9+

Libraries: numpy, pandas, matplotlib

Install dependencies:

pip install numpy pandas matplotlib

2. Run the Platform
python real_time_disaster_platform.py

3. View Outputs

All generated files will be stored in the outputs/ folder.

Open latest_impact_map.png to view the most recent impact hotspots.

📊 Example Outputs

Assets Dataset: 1,500 synthetic points across the Niger Delta region.

Events: 240 simulated events (~4 hours at 1 per minute).

Impact Scores: Severity-labeled, normalized 0–100.

Anomalies: Automatic detection of unusual surges in impact metrics.

📌 Future Extensions

Integration of real hazard feeds (satellite + sensor data)

Streamlit-based real-time dashboard

Multi-hazard modeling (flood, wind, wildfire, landslide)

Asset economic valuation for cost–benefit analysis

📜 License

MIT License — you are free to use, modify, and distribute this code with attribution.

📧 Contact

Developer: Otutu Anslem
github: @Otutu11
