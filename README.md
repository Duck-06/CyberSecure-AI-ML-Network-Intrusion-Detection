# 🛡️ CyberSecure — AI/ML Intrusion Detection
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
## ⛓️ Blockchain-Backed Intrusion Ledger

Our IDS includes a **lightweight custom blockchain** that provides tamper-evident storage for all detected intrusions.  
It is designed specifically for **incident forensics**, **traceability**, and **data integrity assurance**.

---

### 🔧 How It Works

When you run **`launcher.py`**, the following sequence occurs:

1. **`predict_offline.py` runs first**  
   - It checks whether a `chain.json` file already exists in the project directory.  
   - If **no chain exists**, it **creates a new blockchain** and appends a block for every flow in `inputs.csv`.  
   - If a **chain already exists**, it **appends new intrusion blocks** to the existing chain — preserving all previous entries.

2. After predictions finish, **`app.py` starts the Streamlit dashboard**, where users can:
   - View the blockchain contents  
   - Monitor newly added intrusion records  
   - Verify chain integrity  

---

### 🔗 What Each Blockchain Block Contains

Every intrusion is stored as a block containing:

- The model’s verdict (Benign/Intrusion)  
- Intrusion probability & confidence  
- SHA-256 hash of the flow’s features  
- Hash of the previous block  
- Timestamp of the event  
- A small summary of the most important features  
- The block’s final hash (ensuring immutability)  

This creates a **hash-linked ledger**, where each block depends on all previous blocks.

---

### 🔒 Integrity & Tamper Detection

Our blockchain is **fully self-verifying**:

- The GUI includes a **Verify Chain** button connected to `verify_chain.py`  
- It recomputes hashes for every block and checks the entire chain end-to-end  
- If **any external modification** is made — even a single digit in `chain.json`:
  - The chain is immediately flagged as **INVALID**  
  - The system reports the **exact block index** where integrity fails  
  - No new blocks will be appended until the chain is fixed  

This guarantees **forensic-level data integrity**, ideal for SOC environments.

---

### 🔁 Behavior on Multiple Runs

Running **`launcher.py`** multiple times will:

- Keep using the existing `chain.json`  
- Append new blocks for new flows  
- Grow the chain continuously over time  

This enables **long-term accumulation of intrusion history** and helps build a complete audit trail for network security analysis.

---
## 🧠 Explainable AI (XAI) for Transparent Intrusion Decisions

Machine Learning models—especially ensemble methods like XGBoost—often work as **black boxes**, making it difficult for analysts to understand *why* a network flow was flagged as malicious.  
To solve this, our IDS integrates **Explainable AI (XAI)** techniques that bring full transparency to every prediction.

---

### 🔍 Why XAI Is Required in an IDS

Traditional IDS models provide a label (Benign/Intrusion) but fail to answer:

- *Which network features triggered the alert?*  
- *Is this a genuine threat or a false alarm?*  
- *Should the SOC operator escalate, ignore, or investigate further?*  
- *What pattern or behaviour caused the model to classify it as malicious?*

Without explainability, ML-based IDS systems risk becoming **untrusted**, **opaque**, and **difficult to audit**.

XAI solves this by providing **clear, human-interpretable explanations** for every decision, enabling:

- Faster triage during real-time investigation  
- Higher operator trust in the ML model  
- Easier debugging and model improvement  
- Stronger forensic traceability alongside the blockchain ledger  

---

### 🛠️ How XAI Is Implemented in This Project

Our XAI layer uses two major components:

#### **1. SHAP (Shapley Additive Explanations)**
- Provides **per-flow explanations**, showing which features increased or decreased the chance of an intrusion.  
- Generates **force plots & waterfall charts** that visually break down the model's reasoning.  
- Helps analysts understand feature contributions such as:  
  - Flow Duration  
  - Packet Length Variance  
  - Flag Counts  
  - Flow IAT statistics  
  - Header lengths  
- SHAP values reveal *why* the model makes a decision—not just *what* it predicts.

#### **2. Global Feature Importance**
- Gives an overview of which features the model relies on the most across the entire dataset.  
- Helps in optimizing feature engineering and model tuning.  
- Reflects long-term trends in malicious network behaviour.

---

### 🎯 Outcome

By combining **SHAP explainability** with the **Blockchain-based Intrusion Ledger**, the system provides:

- Transparent, trustworthy intrusion decisions  
- Forensic-grade auditability  
- Actionable insights for SOC analysts  
- A clear rationale for every alert  

This XAI integration transforms the IDS from a black-box classifier into a **fully interpretable, analyst-friendly security system**.

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
├── launcher.py
├── prediction_block_summary.csv    # Auto-generated summary
├── verify_chain.py
│
├── data/
│   ├── xgboost_intrusion_model_high_recall.pkl
│   ├── scaler.pkl
│   ├── xgb_high_recall_full_metrics.xlsx
│
├── __pycache__/                    # Auto-generated cache
│   ├── blockchain.cpython-313.pyc
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
| **Reyansh Sakriya** | Machine Learning & XAI Lead |
| **Indraneel Patil** | Blockchain & Security Logic |

> 🤝 A united team effort — Cyber defense requires collaboration 🛡️

---
📄 License

This project is released under the MIT License.
See LICENSE file for full details.

---
📬 Contact Info

👤 Author: [Rudra Tatuskar]
🐙 GitHub: [Duck-06]
🔗 LinkedIn: [Rudra Tatuskar]
