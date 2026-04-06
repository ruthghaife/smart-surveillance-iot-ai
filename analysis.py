"""
Smart Surveillance & IoT Anomaly Detection System
===================================================
AI-powered security intelligence platform covering:
  1. IoT Sensor Anomaly Detection    (multi-class activity classification)
  2. Network Intrusion Detection     (binary + multi-class attack classification)
  3. Time-Series Sensor Visualisation
  4. Model Explainability Dashboard
  5. Real-Time Alert Simulation

Author : Ghaife Mabu Ruth
Contact: ruthghaifemaburuth@gmail.com
TEEP Links: NTNU #1412 (Deep Learning Video Surveillance)
            Chang Gung #956 (ML in Electronic System Reliability)
            NCUE #1288 (AI Applications, Edge Computing, DSP)
            Providence #1116 (Counterfeit Detection with DCNN)
            NTOU #1597 (Autonomous Vehicle, BCI, Heuristic Algorithms)
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.model_selection  import train_test_split, cross_val_score
from sklearn.preprocessing    import LabelEncoder, StandardScaler
from sklearn.ensemble         import (RandomForestClassifier, IsolationForest,
                                       GradientBoostingClassifier)
from sklearn.svm              import SVC
from sklearn.neighbors        import KNeighborsClassifier
from sklearn.linear_model     import LogisticRegression
from sklearn.metrics          import (classification_report, confusion_matrix,
                                       roc_auc_score, roc_curve,
                                       ConfusionMatrixDisplay)
from sklearn.decomposition    import PCA
from sklearn.inspection       import permutation_importance

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE   = os.path.dirname(__file__)
DATA   = os.path.join(BASE, "data")
OUT    = os.path.join(BASE, "outputs")
os.makedirs(OUT, exist_ok=True)

# ── Palette ────────────────────────────────────────────────────────────────────
NAVY   = "#1F3864"
ORANGE = "#C55A11"
TEAL   = "#2E86AB"
GREEN  = "#27AE60"
RED    = "#E74C3C"
YELLOW = "#F39C12"
PURPLE = "#8E44AD"
PALETTE = [NAVY, ORANGE, TEAL, GREEN, RED, YELLOW, PURPLE, "#1ABC9C"]
sns.set_theme(style="whitegrid", font="DejaVu Sans")

ACTIVITY_COLORS = {
    "Normal":          GREEN,
    "Occupied":        TEAL,
    "Intrusion":       RED,
    "Fire_Risk":       ORANGE,
    "Equipment_Fault": PURPLE,
}


# ══════════════════════════════════════════════════════════════════════════════
# 0. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
def load_data():
    gen = os.path.join(DATA, "generate_surveillance_data.py")
    for fname in ["sensor_stream.csv", "network_traffic.csv"]:
        if not os.path.exists(os.path.join(DATA, fname)):
            print("[INFO] Generating surveillance dataset...")
            import subprocess
            subprocess.run([sys.executable, gen], check=True)
            break
    sensors = pd.read_csv(os.path.join(DATA, "sensor_stream.csv"),
                          parse_dates=["timestamp"])
    network = pd.read_csv(os.path.join(DATA, "network_traffic.csv"))
    print(f"[LOAD] Sensor stream: {len(sensors)} records | Network: {len(network)} packets")
    print(f"  Sensor anomalies: {sensors['is_anomaly'].mean()*100:.1f}%")
    print(f"  Network attacks:  {network['is_attack'].mean()*100:.1f}%")
    return sensors, network


# ══════════════════════════════════════════════════════════════════════════════
# 1. IoT TIME-SERIES SENSOR DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
def sensor_dashboard(sensors):
    print("\n[SENSOR] Building IoT sensor time-series dashboard...")

    # Use first 7 days for clarity
    week = sensors.iloc[:168].copy()
    anomaly_mask = week["is_anomaly"] == 1

    fig = plt.figure(figsize=(20, 18), facecolor="#F8F9FA")
    fig.suptitle("Smart Surveillance — IoT Sensor Time-Series Dashboard",
                 fontsize=20, fontweight="bold", color=NAVY, y=0.99)
    gs = gridspec.GridSpec(5, 2, figure=fig, hspace=0.55, wspace=0.3)

    sensor_cols = [
        ("temperature", "Temperature (°C)",  NAVY),
        ("humidity",    "Humidity (%)",       TEAL),
        ("motion",      "Motion Intensity",   ORANGE),
        ("sound_db",    "Sound Level (dB)",   PURPLE),
        ("vibration",   "Vibration Intensity",RED),
    ]

    for idx, (col, ylabel, color) in enumerate(sensor_cols):
        ax = fig.add_subplot(gs[idx, 0])
        ax.plot(week["timestamp"], week[col], color=color, linewidth=1.2, alpha=0.85)

        # Shade anomaly regions
        for i in week.index[anomaly_mask]:
            if i in week.index:
                ax.axvspan(week.loc[i,"timestamp"],
                           week.loc[min(i+1, week.index[-1]),"timestamp"],
                           alpha=0.2, color=RED, linewidth=0)

        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(f"{ylabel} — 7-Day Trace (Red=Anomaly)", fontweight="bold",
                     color=NAVY, fontsize=10)
        ax.tick_params(axis="x", labelrotation=30, labelsize=7)

    # Activity distribution
    ax = fig.add_subplot(gs[0:2, 1])
    act_counts = sensors["activity"].value_counts()
    colors_act = [ACTIVITY_COLORS.get(a, NAVY) for a in act_counts.index]
    wedges, texts, auts = ax.pie(act_counts.values, labels=act_counts.index,
                                  colors=colors_act, autopct="%1.1f%%", startangle=90)
    for at in auts:
        at.set_fontsize(9)
        at.set_fontweight("bold")
    ax.set_title("Activity Label Distribution (30 Days)", fontweight="bold", color=NAVY)

    # Hourly anomaly rate
    ax = fig.add_subplot(gs[2:4, 1])
    hourly_anom = sensors.groupby("hour")["is_anomaly"].mean() * 100
    colors_h = [RED if hourly_anom[h] > hourly_anom.mean() else NAVY for h in hourly_anom.index]
    ax.bar(hourly_anom.index, hourly_anom.values, color=colors_h, edgecolor="white")
    ax.axhline(hourly_anom.mean(), color=ORANGE, linewidth=2, linestyle="--",
               label=f"Mean={hourly_anom.mean():.1f}%")
    ax.set_title("Anomaly Rate by Hour of Day", fontweight="bold", color=NAVY)
    ax.set_xlabel("Hour")
    ax.set_ylabel("Anomaly Rate (%)")
    ax.set_xticks(range(0, 24))
    ax.legend()

    # Sensor correlation
    ax = fig.add_subplot(gs[4, :])
    corr = sensors[["temperature","humidity","motion","sound_db","vibration","is_anomaly"]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                ax=ax, linewidths=0.5, cbar_kws={"shrink":0.6}, annot_kws={"size":9})
    ax.set_title("Sensor Feature Correlation Matrix", fontweight="bold", color=NAVY)

    path = os.path.join(OUT, "01_sensor_dashboard.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. ACTIVITY CLASSIFICATION (Multi-class)
# ══════════════════════════════════════════════════════════════════════════════
def activity_classification(sensors):
    print("\n[ACTIVITY] Training multi-class activity classifier...")

    FEATURES = ["temperature","humidity","motion","sound_db","vibration",
                "hour","day_of_week","is_night"]
    le = LabelEncoder()
    y  = le.fit_transform(sensors["activity"])
    X  = sensors[FEATURES]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        "Random Forest":     RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, learning_rate=0.08, random_state=42),
        "K-NN (k=5)":        KNeighborsClassifier(n_neighbors=5),
    }

    best_acc, best_name, best_clf, best_pred = 0, "", None, None
    for name, clf in models.items():
        clf.fit(X_train, y_train)
        pred  = clf.predict(X_test)
        acc   = (pred == y_test).mean()
        cv    = cross_val_score(clf, X_train, y_train, cv=5, scoring="accuracy").mean()
        if acc > best_acc:
            best_acc, best_name, best_clf, best_pred = acc, name, clf, pred
        print(f"  {name:<24}: Accuracy={acc:.3f} | CV Acc={cv:.3f}")

    class_names = le.classes_
    print(f"  → Best: {best_name} ({best_acc:.3f})")
    print(f"\n{classification_report(y_test, best_pred, target_names=class_names)}")

    # Feature importance
    fi = pd.Series(best_clf.feature_importances_, index=FEATURES).sort_values()

    # PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    # ── Figure 2: Activity Classification Dashboard ──────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), facecolor="#F8F9FA")
    fig.suptitle(f"Activity Classification — {best_name} (Accuracy={best_acc:.3f})",
                 fontsize=18, fontweight="bold", color=NAVY)

    # Confusion matrix
    ax = axes[0,0]
    cm = confusion_matrix(y_test, best_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, cbar=False, linewidths=0.5)
    ax.set_title("Confusion Matrix", fontweight="bold", color=NAVY)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)

    # Feature importance
    ax = axes[0,1]
    fi_colors = [ORANGE if fi.index[-1] == f else NAVY for f in fi.index]
    fi.plot(kind="barh", ax=ax, color=fi_colors, edgecolor="white")
    ax.set_title("Feature Importance", fontweight="bold", color=NAVY)
    ax.set_xlabel("Importance")

    # PCA scatter
    ax = axes[0,2]
    for i, act in enumerate(class_names):
        mask = (sensors["activity"] == act).values
        ax.scatter(X_pca[mask,0], X_pca[mask,1], label=act, alpha=0.4,
                   s=18, color=ACTIVITY_COLORS.get(act, PALETTE[i]))
    ax.set_title("PCA Projection by Activity", fontweight="bold", color=NAVY)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.legend(fontsize=8)

    # Motion vs Sound coloured by activity
    ax = axes[1,0]
    for i, act in enumerate(class_names):
        subset = sensors[sensors["activity"] == act]
        ax.scatter(subset["motion"], subset["sound_db"], label=act, alpha=0.35,
                   s=15, color=ACTIVITY_COLORS.get(act, PALETTE[i]))
    ax.set_title("Motion vs Sound Level", fontweight="bold", color=NAVY)
    ax.set_xlabel("Motion Intensity")
    ax.set_ylabel("Sound (dB)")
    ax.legend(fontsize=8)

    # Temperature distribution per activity
    ax = axes[1,1]
    for i, act in enumerate(class_names):
        subset = sensors.loc[sensors["activity"]==act, "temperature"]
        ax.hist(subset, bins=20, alpha=0.6, label=act, density=True,
                color=ACTIVITY_COLORS.get(act, PALETTE[i]), edgecolor="white")
    ax.set_title("Temperature Distribution per Activity", fontweight="bold", color=NAVY)
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)

    # Vibration vs Motion by anomaly
    ax = axes[1,2]
    ax.scatter(sensors.loc[sensors["is_anomaly"]==0,"vibration"],
               sensors.loc[sensors["is_anomaly"]==0,"motion"],
               alpha=0.3, s=14, color=GREEN, label="Normal")
    ax.scatter(sensors.loc[sensors["is_anomaly"]==1,"vibration"],
               sensors.loc[sensors["is_anomaly"]==1,"motion"],
               alpha=0.5, s=22, color=RED, label="Anomaly", marker="^")
    ax.set_title("Vibration vs Motion (Anomaly Highlighted)", fontweight="bold", color=NAVY)
    ax.set_xlabel("Vibration Intensity")
    ax.set_ylabel("Motion Intensity")
    ax.legend()

    path = os.path.join(OUT, "02_activity_classification.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. NETWORK INTRUSION DETECTION
# ══════════════════════════════════════════════════════════════════════════════
def intrusion_detection(network):
    print("\n[INTRUSION] Training network intrusion detection classifier...")

    FEATURES = ["packet_size","duration","n_packets","byte_rate",
                "flag_syn","flag_rst","dst_port","login_attempts"]
    le = LabelEncoder()
    y  = le.fit_transform(network["attack_type"])
    X  = network[FEATURES]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)

    # Anomaly detection (unsupervised)
    iso = IsolationForest(n_estimators=200, contamination=0.30, random_state=42)
    iso.fit(X_tr_s)
    iso_pred = iso.predict(X_te_s)   # -1 = anomaly, 1 = normal
    iso_anom = (iso_pred == -1).astype(int)
    y_binary = (y_test > 0).astype(int)    # normal=0 vs any attack=1
    iso_acc  = (iso_anom == y_binary).mean()
    print(f"  Isolation Forest (unsupervised) binary accuracy: {iso_acc:.3f}")

    # Supervised multi-class
    rf = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_acc  = (rf_pred == y_test).mean()
    rf_proba = rf.predict_proba(X_test)
    class_names = le.classes_
    print(f"  Random Forest (supervised) accuracy: {rf_acc:.3f}")
    print(f"\n{classification_report(y_test, rf_pred, target_names=class_names)}")

    # Feature importance
    fi = pd.Series(rf.feature_importances_, index=FEATURES).sort_values()

    # ── Figure 3: Intrusion Detection Dashboard ─────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), facecolor="#F8F9FA")
    fig.suptitle(f"Network Intrusion Detection — RF Accuracy={rf_acc:.3f}",
                 fontsize=18, fontweight="bold", color=NAVY)

    # Confusion matrix
    ax = axes[0,0]
    cm = confusion_matrix(y_test, rf_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges",
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, cbar=False, linewidths=0.5)
    ax.set_title("Multi-Class Confusion Matrix", fontweight="bold", color=NAVY)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right", fontsize=8)

    # Feature importance
    ax = axes[0,1]
    fi_colors = [ORANGE if fi.index[-1] == f else NAVY for f in fi.index]
    fi.plot(kind="barh", ax=ax, color=fi_colors, edgecolor="white")
    ax.set_title("Feature Importance (Random Forest)", fontweight="bold", color=NAVY)

    # Byte rate by attack type
    ax = axes[0,2]
    for i, att in enumerate(class_names):
        subset = network.loc[network["attack_type"]==att, "byte_rate"]
        ax.hist(subset, bins=20, alpha=0.65, label=att, density=True,
                color=PALETTE[i % len(PALETTE)], edgecolor="white")
    ax.set_title("Byte Rate Distribution by Attack Type", fontweight="bold", color=NAVY)
    ax.set_xlabel("Byte Rate (bytes/s)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)

    # Packet size vs n_packets scatter
    ax = axes[1,0]
    for i, att in enumerate(class_names):
        subset = network[network["attack_type"]==att]
        ax.scatter(subset["packet_size"], subset["n_packets"],
                   alpha=0.4, s=16, label=att, color=PALETTE[i % len(PALETTE)])
    ax.set_title("Packet Size vs # Packets", fontweight="bold", color=NAVY)
    ax.set_xlabel("Packet Size (bytes)")
    ax.set_ylabel("# Packets in Flow")
    ax.set_yscale("log")
    ax.legend(fontsize=8)

    # Isolation Forest score distribution
    ax = axes[1,1]
    scores = iso.decision_function(X_te_s)
    ax.hist(scores[y_binary==0], bins=30, color=GREEN, alpha=0.7,
            label="Normal", edgecolor="white")
    ax.hist(scores[y_binary==1], bins=30, color=RED, alpha=0.7,
            label="Attack", edgecolor="white")
    ax.axvline(0, color="black", linewidth=2, linestyle="--", label="Decision boundary")
    ax.set_title("Isolation Forest Anomaly Scores", fontweight="bold", color=NAVY)
    ax.set_xlabel("Anomaly Score (lower=more anomalous)")
    ax.set_ylabel("Count")
    ax.legend()

    # Attack type distribution
    ax = axes[1,2]
    att_counts = network["attack_type"].value_counts()
    colors_att = [PALETTE[i % len(PALETTE)] for i in range(len(att_counts))]
    ax.bar(att_counts.index, att_counts.values, color=colors_att, edgecolor="white")
    ax.set_title("Attack Type Distribution", fontweight="bold", color=NAVY)
    ax.set_ylabel("Count")
    ax.set_xticklabels(att_counts.index, rotation=15, ha="right")
    for i, v in enumerate(att_counts.values):
        ax.text(i, v + 10, str(v), ha="center", fontweight="bold", fontsize=9)

    path = os.path.join(OUT, "03_intrusion_detection.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. ALERT SIMULATION DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
def alert_simulation(sensors):
    print("\n[ALERT] Simulating real-time anomaly alert system...")

    # Use first 7 days
    week = sensors.iloc[:168].copy().reset_index(drop=True)
    week["alert_level"] = "OK"
    week.loc[week["activity"]=="Intrusion",      "alert_level"] = "CRITICAL"
    week.loc[week["activity"]=="Fire_Risk",       "alert_level"] = "CRITICAL"
    week.loc[week["activity"]=="Equipment_Fault", "alert_level"] = "WARNING"
    week.loc[week["activity"]=="Occupied",        "alert_level"] = "INFO"

    alert_colors = {"OK": GREEN, "INFO": TEAL, "WARNING": ORANGE, "CRITICAL": RED}

    fig, axes = plt.subplots(3, 1, figsize=(18, 14), facecolor="#0D1117")
    fig.suptitle("🔴 SMART SURVEILLANCE — REAL-TIME ALERT DASHBOARD",
                 fontsize=18, fontweight="bold", color="white", y=0.99)

    for ax in axes:
        ax.set_facecolor("#0D1117")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")

    # Temperature alert trace
    ax = axes[0]
    ax.plot(week.index, week["temperature"], color="#00BFFF", linewidth=1.5, alpha=0.9)
    for _, row in week[week["alert_level"]=="CRITICAL"].iterrows():
        ax.axvline(row.name, color=RED, alpha=0.4, linewidth=1)
    for _, row in week[week["alert_level"]=="WARNING"].iterrows():
        ax.axvline(row.name, color=ORANGE, alpha=0.3, linewidth=1)
    ax.axhline(35, color=RED, linewidth=1.5, linestyle="--", alpha=0.8, label="Fire threshold (35°C)")
    ax.set_ylabel("Temperature (°C)", color="white")
    ax.set_title("Temperature Sensor — Alert Overlay", color="white", fontweight="bold")
    ax.legend(facecolor="#1A1A2E", labelcolor="white", fontsize=9)
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")

    # Motion + Vibration dual trace
    ax = axes[1]
    ax2 = ax.twinx()
    ax.plot(week.index, week["motion"],    color=ORANGE, linewidth=1.5, label="Motion", alpha=0.85)
    ax2.plot(week.index, week["vibration"],color=PURPLE, linewidth=1.2, label="Vibration", alpha=0.85)
    for _, row in week[week["alert_level"]=="CRITICAL"].iterrows():
        ax.axvspan(max(0,row.name-0.5), min(len(week)-1,row.name+0.5), alpha=0.2, color=RED)
    ax.set_ylabel("Motion", color=ORANGE)
    ax2.set_ylabel("Vibration", color=PURPLE)
    ax.set_title("Motion & Vibration — Anomaly Spikes Highlighted (Red)", color="white", fontweight="bold")
    ax.set_facecolor("#0D1117")
    ax2.set_facecolor("#0D1117")
    ax.tick_params(colors="white")
    ax2.tick_params(colors="white")
    for spine in ax.spines.values(): spine.set_edgecolor("#333333")
    for spine in ax2.spines.values(): spine.set_edgecolor("#333333")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1+lines2, labels1+labels2, facecolor="#1A1A2E", labelcolor="white", fontsize=9)

    # Alert level timeline
    ax = axes[2]
    level_map = {"OK": 0, "INFO": 1, "WARNING": 2, "CRITICAL": 3}
    week["alert_num"] = week["alert_level"].map(level_map)
    for level, color in alert_colors.items():
        mask = week["alert_level"] == level
        ax.scatter(week.index[mask], week.loc[mask,"alert_num"],
                   color=color, s=35, alpha=0.85, label=level, zorder=3)
    ax.set_yticks([0,1,2,3])
    ax.set_yticklabels(["OK","INFO","WARNING","CRITICAL"], color="white")
    ax.set_xlabel("Hour Index (7 Days)", color="white")
    ax.set_title("Alert Level Timeline — 7-Day Surveillance Window", color="white", fontweight="bold")
    ax.set_facecolor("#0D1117")
    ax.tick_params(colors="white")
    for spine in ax.spines.values(): spine.set_edgecolor("#333333")
    ax.legend(facecolor="#1A1A2E", labelcolor="white", fontsize=9, ncol=4)

    path = os.path.join(OUT, "04_alert_dashboard.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0D1117")
    plt.close()
    print(f"  → Saved: {path}")

    # Print alert summary
    summary = week["alert_level"].value_counts()
    print("\n  ALERT SUMMARY (7-day window):")
    for lvl in ["CRITICAL","WARNING","INFO","OK"]:
        if lvl in summary:
            print(f"  {'🔴' if lvl=='CRITICAL' else '🟡' if lvl=='WARNING' else '🔵' if lvl=='INFO' else '🟢'} {lvl}: {summary[lvl]} events")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 62)
    print("  SMART SURVEILLANCE & IoT ANOMALY DETECTION SYSTEM")
    print("  Author: Ghaife Mabu Ruth | ruthghaifemaburuth@gmail.com")
    print("=" * 62)

    import subprocess
    gen_path = os.path.join(DATA, "generate_surveillance_data.py")
    for f in ["sensor_stream.csv","network_traffic.csv"]:
        if not os.path.exists(os.path.join(DATA, f)):
            subprocess.run([sys.executable, gen_path], check=True)
            break

    sensors, network = load_data()
    sensor_dashboard(sensors)
    activity_classification(sensors)
    intrusion_detection(network)
    alert_simulation(sensors)

    print("\n" + "=" * 62)
    print("  ALL OUTPUTS SAVED → outputs/")
    print("  01_sensor_dashboard.png")
    print("  02_activity_classification.png")
    print("  03_intrusion_detection.png")
    print("  04_alert_dashboard.png")
    print("=" * 62)
