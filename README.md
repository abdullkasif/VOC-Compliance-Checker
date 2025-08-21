# VOC Emission Compliance Checker  

A research-oriented project to analyze the **QM9 molecular dataset** for VOC (Volatile Organic Compound) compliance under the **German TA Luft (§11)** regulations.  
The system processes molecular data, extracts physicochemical properties, and prepares groundwork for predicting vapor pressure thresholds relevant to VOC classification.  

---

## Task Progress  
We are following a structured **20-task roadmap**. Below is the current progress:  

- [x] **Task 1** – Environment setup  
- [x] **Task 2** – QM9 dataset download & subset creation (VOC-compliant molecules only)  
- [x] **Task 3** – VRAM footprint analysis  
- [x] **Task 4** – Data inspection & target properties documentation  
- [x] **Task 5** – Identification of VOC threshold in German TA Luft (§11)  
- [ ] **Task 6** – GitHub repo setup (current task)  
- [ ] **Task 7–20** – Model development, training, evaluation, demo interface, and deployment  

---

## Setup Instructions  

Clone the repository:  
```bash
git clone https://github.com/abdullkasif/VOC-Compliance-Checker.git
cd VOC-Compliance-Checker
```

Create and activate a Python environment (recommended: Python 3.10+):  
```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

Install dependencies:  
```bash
pip install -r requirements.txt
```

Run dataset download (Task 2) (which is already present inside data/):  
```bash
python src/download_qm9.py
```

Inspect data (Task 4):  
```bash
python src/data_inspection.py
```

---

##  Repository Structure  

```
VOC-Compliance-Checker/
├── data/              # QM9 dataset (raw, processed, subsets)
├── src/               # Python source code
├── logs/              # Task-wise documentation
|   └──proof/          # Proof images for completed tasks     
├── demo/              # Placeholder for demos
├── VOC_threshold.md   # VVOC Vapor Pressure Threshold in German Regulations
├── README.md          # Project overview (this file)
└── requirements.txt   # Python dependencies
```

---

##  Regulatory Context  

VOC compliance in this project is based on:  
👉 [VOC_threshold.md](./VOC_threshold.md)  

This document outlines the **exact vapor pressure thresholds** defined in German **TA Luft regulations (§11)** for identifying VOCs.  

---

✨ This repository is part of a step-by-step learning and implementation journey toward building an AI-powered VOC compliance checker.  