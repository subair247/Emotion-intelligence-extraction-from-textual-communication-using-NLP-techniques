# TECHNICAL PROJECT DOCUMENTATION & README
## Project Title: Emotion Intelligence Extraction from Textual Communication Using NLP Techniques

---

### 1. Document Overview
This document serves as the comprehensive technical documentation and system architecture manual for the final-year capstone project: **Emotion Intelligence Extraction from Textual Communication**. It covers system configuration, structural layouts, execution guidelines, and implementation workflows designed for both evaluation panels and engineering handoffs.

---

### 2. Project Abstract
In modern digital communications, identifying the underlying emotional intelligence context—such as empathy, sentiment intensity, and cognitive state—is crucial for mental health risk assessments, automated customer success auditing, and human-computer interactions. 

This project delivers an end-to-end Natural Language Processing (NLP) pipeline that ingests raw textual data, processes semantic representations, and extracts multi-class emotional vectors. The core engine utilizes a hybrid dual-model engineering architecture, implementing a primary deep-learning transformer pipeline alongside an optimized machine learning ensemble fallback layer to guarantee high system availability and execution reliability.

---

### 3. Core Architectural Highlights
* **Hybrid Dual-Method Engine:** Integrates a state-of-the-art Hugging Face transformer pipeline optimized for contextual understanding, with a standalone TF-IDF vectorization + Ensemble Machine Learning model (Scikit-Learn) serving as a robust native fallback layer.
* **Streamlined UI Execution:** Features an interactive, low-latency web workspace engineered via **Streamlit** for real-time inference, text analytics visualization, and performance breakdown reporting.
* **Isolated Dependency Design:** Engineered explicitly inside a dedicated virtual environment (`venv`) to prevent scope pollution, ensuring deterministic reproducibility.

---

### 4. Repository & File Structure
As configured within the workspace environment, the project maintains the following standard hierarchical layout:
```text
BATCH 9 HABEEB Emotion intelligence extraction from textual communication using NLP techniques (G9)/
├── venv/                           # Isolated Python Virtual Environment
├── emotion_model_v2/               # Core extracted directory containing serialized models/weights
├── app.py                          # Streamlit main application file and orchestration script
├── emotion_eda.png                 # Exploratory Data Analysis chart assets
├── emotion_results.png             # Visual matrix of model evaluation performance results
├── Emotion_To_Text.ipynb           # Jupyter Notebook for experimental NLP prototyping and training
├── requirements.txt                # Unified Python deployment package dependencies
└── README.md                       # Active repository documentation
```

---

### 5. Dependency Manifest (`requirements.txt`)
The execution layer requires the following locked foundational packages:
* `streamlit`: Interactive dashboard construction.
* `transformers`: Interface for Hugging Face pre-trained model inference pipelines.
* `torch`: Computational backend tensor engine for neural execution paths.
* `scikit-learn`: Feature extraction (TF-IDF) and ensemble evaluation metrics.
* `plotly`: High-fidelity interactive charts for sentiment distributions.
* `pandas` & `numpy`: High-performance structural data manipulation frameworks.
* `joblib` & `scipy`: Serialized mathematical optimization utilities.

---

### 6. Environment Provisioning & Installation Steps

#### Step 1: Initialize Shell Environment Policies
To allow execution of the isolated environment startup script within Windows PowerShell environments, configure the system execution scope with remote signing privileges:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned```

#### Step 2: Activate the Isolated Environment
Navigate to your project root workspace directory and run the native activation script to target your local environment:
```powershell
& ".\venv\Scripts\Activate.ps1"
```

#### Step 3: Verify and Synchronize Package Dependencies
Ensure all foundational calculation engines match the target requirements:
```powershell
pip install -r requirements.txt
```

#### Step 4: Run the Production Web Application
Orchestrate and spawn the web presentation framework on your local loopback address:
```powershell
streamlit run app.py
```

---

### 7. Core Workflow Pipeline
1.  **Ingestion:** Text is submitted through the interactive Streamlit textarea component.
2.  **Preprocessing:** Sequences are normalized, tokenized, and stripped of linguistic noise.
3.  **Inference Routing:**
    * The application attempts loading the optimized **Hugging Face pre-trained pipeline**.
    * If offline or memory-restricted, the system gracefully intercepts and down-routes processing to the pre-compiled **TF-IDF + Ensemble Model** folder infrastructure (`emotion_model_v2`).
4.  **Analytics Layer:** Prediction matrices are broken down visually into probability arrays using **Plotly** tables and trend graphs before rendering to the client viewport.

---
*Document prepared for B.Tech Final Year Academic Evaluation - Specialization in Artificial Intelligence and Data Science.*