# Smart Surveillance & IoT Anomaly Detection System

> **Author:** Ghaife Mabu Ruth | ruthghaifemaburuth@gmail.com  
> **Institution:** GNA University, Phagwara, India (B.Sc. Information Technology)  
> **Domains Covered:** AI Surveillance · Edge Computing · IoT Sensor Analytics · Network Security · Anomaly Detection

---

##  Project Overview

An end-to-end AI-powered smart surveillance and security intelligence platform that processes multi-channel IoT sensor streams and network traffic data to detect anomalies, classify activity events, and simulate real-time alert systems. The system covers both **physical security** (sensor-based) and **cyber security** (network intrusion detection).

---

##  What This Project Demonstrates

| Module | Task | Output |
|--------|------|--------|
| Sensor Dashboard | 5-channel IoT time-series EDA + anomaly overlay | 7-day trace with alert shading |
| Activity Classification | Multi-class (5 classes): Normal, Occupied, Intrusion, Fire Risk, Equipment Fault | Accuracy ~90%+ |
| Intrusion Detection | Network attack classification (DoS, Port Scan, Brute Force, Data Exfil) + Isolation Forest | AUC / multi-class report |
| Alert Dashboard | Real-time CRITICAL/WARNING/INFO/OK event timeline | Dark-mode surveillance UI |

---

##  Project Structure

```
project3_smart_surveillance/
│
├── data/
│   ├── generate_surveillance_data.py  ← Generates IoT + network datasets
│   ├── sensor_stream.csv              ← 720 hourly sensor readings (30 days)
│   └── network_traffic.csv           ← 5,000 network packet records
│
├── analysis.py                        ← MAIN FILE — full pipeline
│
├── outputs/
│   ├── 01_sensor_dashboard.png        ← Time-series + correlation
│   ├── 02_activity_classification.png ← Multi-class ML results
│   ├── 03_intrusion_detection.png     ← Network IDS results
│   └── 04_alert_dashboard.png         ← Dark-mode real-time alert view
│
├── requirements.txt
└── README.md
```

---

##  Quick Start

```bash
git clone https://github.com/ruthghaife/smart-surveillance-iot-ai.git
cd smart-surveillance-iot-ai/project3_smart_surveillance

pip install -r requirements.txt

python data/generate_surveillance_data.py
python analysis.py
```

---

##  Key Results

- **Activity Classification Accuracy:** ~93% (Random Forest on 5 classes)
- **Intrusion Detection:** RF multi-class accuracy ~94% | Isolation Forest binary baseline
- **Anomaly Rate:** ~15% of sensor readings flagged (Intrusion + Fire + Equipment Fault)
- **Peak Anomaly Hours:** 22:00–05:00 (night watch window)
- **Top Network Features:** byte_rate, login_attempts, n_packets

---

##  Tech Stack

```
Python 3.x · Scikit-learn (RF, GB, IsolationForest, KNN, PCA) · Pandas · NumPy · Matplotlib · Seaborn
```

---

##  Contact

**Ghaife Mabu Ruth**  
ruthghaifemaburuth@gmail.com | +91 9501385794  
🇨🇲 Cameroon → 🇮🇳 GNA University, India 
