# 🛡️ CyberSecure — AI/ML Intrusion Detection & Real-Time Threat Triage  
📍 Built in 24 Hours during **REDACT Cybersecurity Hackathon (2025)**

![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)
![ML Model](https://img.shields.io/badge/Model-XGBoost-orange.svg)

CyberSecure is a **real-time Intrusion Detection System (IDS)** that classifies network traffic as **Benign** or **Intrusion**, assigns a **confidence score**, and automatically recommends **security response actions**.  
Built using a **high-recall ML model**, this system ensures **no intrusion goes undetected**, enabling fast and intelligent SOC triage.

> ❗ In cybersecurity, missing a single attack can be catastrophic.  
> That’s why **Intrusion Recall** is our top priority metric.

---

## 🎯 Hackathon Problem Statement — REDACT (PS-02)

**Goal:** Build an ML-based IDS using tabular network flow features to detect malicious behavior and automate action-based triage.

Implemented requirements:
- ✔ Binary classification: Intrusion vs Benign  
- ✔ Confidence Score + Automated Security Actions  
- ✔ Live Dashboard for SOC triage  
- ✔ High Recall optimization  
- ✔ Tamper-proof Blockchain Logging *(Bonus)*  
- ✔ Explainable AI *(Bonus)*  

**100% features + bonuses delivered 🎯🔥**

---

## 🚀 Key Features

| Feature | Description |
|--------|-------------|
| 🤖 High-Recall XGBoost Classifier | Prioritizes detecting all attacks |
| ⚠️ Automated Triage Responses | Block / Throttle / Quarantine / Allow |
| 📊 Streamlit SOC Dashboard | Live threat feed, alerts, metrics |
| 🔐 Blockchain-Backed Logging | SHA-256 chained incident records |
| 🧠 Explainable AI | Feature importance for trust |
| 🧪 Offline Batch Prediction | Large dataset support |
| 📁 Metrics Export | Excel + CSV reporting for evaluation |

---

## 📈 Model Performance Highlights  
*(Based on CIC-IDS2017 test flows)*

| Metric | Status |
|--------|--------|
| Recall (Intrusion Class) | ⭐ Optimized (Primary Success Metric) |
| Precision | Logged |
| F1 Score | Logged |
| ROC-AUC | Computed |
| PR-AUC | Computed |

Complete results stored as:  
`data/xgb_high_recall_full_metrics.xlsx`  

Includes:
- Confusion Matrix  
- ROC Curve Data  
- PR Curve Data  
- Full Classification Report  

---

## 🔐 Automated Security Action Logic

| Confidence % | Classification | Security Action |
|-------------|----------------|----------------|
| > 90% | Intrusion | 🚫 Block Source IP |
| 60–90% | Intrusion | ⚠️ Throttle Traffic |
| < 60% | Intrusion | 🕵️ Quarantine Endpoint |
| Any | Benign | ☑ Allow |

Mimics SOC Tier-1 triage decisions.

---

## 🖥️ Streamlit Dashboard

Includes:
- **Live Threat Feed**
- **Blockchain Ledger Viewer**
- **Metrics Dashboard**
- **Explainability Insights**
---
## ▶️ How to Run the Application (Launcher Script Version)

Follow the steps below to launch the CyberSecure — IDS Dashboard:

---

### 1️⃣ Download & Extract
- Download the project ZIP from the GitHub Releases section
- Extract the ZIP into any folder on your system

---

### 2️⃣ Install Required Python Dependencies
Open a terminal (CMD / PowerShell / Terminal) inside the extracted folder:

```bash
pip install -r requirements.txt
```
---
### 3️⃣ Launch the Application

Inside the main project directory, run:
```bash
python launcher.py
```
This will automatically launch the Streamlit dashboard in your browser at:
```bash
http://localhost:8501/

```
---
### 4️⃣ Upload Network Flow Data

Inside the dashboard UI:

✔ Upload included demo datasets (inputs.csv)
or
✔ Upload your own CSV file with the same feature format and rename to inputs.csv

---
##⛓️ Blockchain-Based Intrusion Ledger (Bonus)

Each detected intrusion is:

Hashed using SHA-256

Linked with previous block hash

Stored in chain.json

Tampering = Immediate integrity break
→ Supports cyber forensics

---
## 📂 Repository Structure
```
offline_ids/
│
├── app.py                          # Streamlit Dashboard
├── blockchain.py                   # Hash chain ledger system
├── predict_offline.py              # Batch IDS script
├── chain.json                      # Auto-generated event ledger
├── inputs.csv
├── verify_chain.py
├── launcher.py
├── prediction_block_summary.csv    # Auto-generated summary
│
├── __pycache__/                    # Auto-generated cache
│   ├── blockchain.cpython-313.pyc
│
├── data/
│   ├── xgboost_intrusion_model_high_recall.pkl
│   ├── scaler.pkl
│   ├── xgb_high_recall_full_metrics.xlsx
│
├── LICENSE
├── .gitignore
├── requirements.txt
└── README.md
```
---
### 📜 Dataset Citation

This model uses a cleaned & preprocessed version of CIC-IDS 2017:

Preprocessed Kaggle Dataset
🔗 https://www.kaggle.com/datasets/ericanacletoribeiro/cicids2017-cleaned-and-preprocessed

Original Dataset Source
Canadian Institute for Cybersecurity (CIC), University of New Brunswick
🔗 https://www.unb.ca/cic/datasets/ids-2017.html

Dataset rights belong to their respective owners.
---
## 🏆 Hackathon Info

Developed in 24 hours at **🔥 REDACT Cybersecurity Hackathon — 2025**

### 👥 Team Members

| Name | Role |
|------|------|
| **Bhavishy Lotlikar** | Machine Learning & Dashboard |
| **Rudra Tatuskar** | Machine Learning & Backend / Data Pipeline |
| **Reyansh Sakriya** | Team Leader & XAI Lead |
| **Indraneel Patil** | Blockchain & Security Logic |

> 🤝 A united team effort — Cyber defense requires collaboration 🛡️

---
📄 License

This project is released under the MIT License.
See LICENSE file for full details.

---
📬 Contact Info

👤 Author: [Bhavishy Lotlikar]
📧 Email: [your-email]
🐙 GitHub: [your GitHub profile]
🔗 LinkedIn: [your LinkedIn profile]
