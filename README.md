
# Optimizing Diabetes Risk Screening for Capacity-Constrained Healthcare Systems Using Boosting Models

**National Economics University (NEU) | College of Technology | Faculty of Data Science and Artificial Intelligence**

**Data Mining – Group 9 | November 2026**

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![Streamlit](https://img.shields.io/badge/Deployment-Streamlit-red?logo=streamlit)
![Status](https://img.shields.io/badge/Status-Research%20Prototype-green)

> **Disclaimer:** This project is an academic research prototype for diabetes-status screening and capacity-based patient prioritization. It is not intended for clinical diagnosis or medical decision-making.

---

## 1. Project Overview

Healthcare systems frequently face resource constraints that limit the number of individuals who can receive follow-up screening. Under these conditions, traditional classification metrics such as accuracy and ROC-AUC may not fully reflect the practical objective of identifying the highest-risk individuals within a limited screening capacity.

This project develops and evaluates a **capacity-constrained diabetes risk screening framework** using machine learning, focusing on the effectiveness of patient risk ranking at different healthcare capacity levels.

The study uses the **CDC Diabetes Health Indicators dataset**, derived from the 2015 Behavioral Risk Factor Surveillance System (BRFSS), containing:

- **253,680 observations**
- **21 predictor variables**
- **35,346 positive diabetes-status observations (13.93%)**
- **218,334 negative observations (86.07%)**

The central investigation compares two XGBoost optimization strategies:

**Global Optimization**

A single XGBoost model is optimized using PR-AUC and evaluated across all healthcare capacity levels.

**Capacity-Specific Optimization**

Six separate XGBoost models are optimized directly for Precision@K at capacities of 0.5%, 1%, 2.5%, 5%, 10%, and 20%.

The project further integrates model interpretability, a Streamlit screening prototype, and an offline monitoring and retraining framework.

### Key Finding

**Capacity-specific optimization did not consistently outperform global optimization on the locked test set.**

The global XGBoost model performed better at four of six capacities, tied at one capacity, and was only marginally worse at another.

These findings support using a single globally optimized model for flexible patient prioritization while avoiding the additional complexity of maintaining six specialized models.

---

## 2. Research Questions

**Primary Research Question**

How effectively can survey-based health and behavioral indicators be used to predict diabetes risk under different healthcare capacity constraints?

**RQ1:** Which health, behavioral, demographic, and socioeconomic factors are most strongly associated with diabetes risk?

**RQ2:** How effectively can machine-learning models identify high-risk individuals across different healthcare capacity levels?

**RQ3:** Does capacity-specific optimization improve Precision@K compared with a globally optimized model?

**RQ4:** How can the resulting model be translated into an interpretable and deployable screening workflow for capacity-constrained healthcare settings?

---

## 3. Dataset

### 3.1 Data Source

The research uses the CDC Diabetes Health Indicators dataset from the UCI Machine Learning Repository.

**Official dataset:**

https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators

**Dataset backup in this repository:**

[diabetes_binary_health_indicators_BRFSS2015.csv](diabetes_binary_health_indicators_BRFSS2015.csv)

### 3.2 Dataset Characteristics

| Property | Description |
|---|---|
| Source | CDC BRFSS 2015 |
| Observations | 253,680 |
| Predictors | 21 |
| Target | Diabetes_binary |
| Problem type | Binary classification |
| Positive observations | 35,346 |
| Positive prevalence | 13.93% |
| Missing values | None detected |

The predictor variables describe health conditions, health behaviors, demographic characteristics, socioeconomic status, and healthcare access.

Continuous variables include BMI, MentHlth, and PhysHlth. Ordinal variables include GenHlth, Age, Education, and Income. Other predictors are primarily binary indicators.

### 3.3 Data Preparation

The preprocessing workflow includes:

1. Data acquisition and validation.
2. Missing-value and invalid-range checks.
3. Duplicate and outlier examination.
4. Exploratory data analysis.
5. Feature and target separation.
6. Stratified train-test splitting.
7. Model-specific preprocessing.

Identical records and valid-range outliers were retained because there was insufficient evidence to classify them as erroneous observations.

Standardization was applied to Logistic Regression, while boosting models used the original feature scales.

---

## 4. Research Methodology

The research follows a comparative machine-learning framework.

### Stage 1 – Exploratory Data Analysis

EDA investigates class imbalance, feature distributions, diabetes-status associations, and relevant health indicators.

### Stage 2 – Baseline and Model Screening

Logistic Regression serves as the baseline.

Four boosting algorithms are screened:

- HistGradientBoosting
- XGBoost
- LightGBM
- CatBoost

Following initial screening and comparative tuning, XGBoost is selected for deeper optimization.

### Stage 3 – Model Optimization

Two optimization approaches are evaluated.

| Component | Global Optimization | Capacity-Specific Optimization |
|---|---|---|
| Algorithm | XGBoost | XGBoost |
| Objective | Mean CV PR-AUC | Mean CV Precision@K |
| Optuna search | 30 pilot + 100 full trials | 30 trials per capacity |
| Number of final models | 1 | 6 |
| Evaluation capacities | All six | Corresponding capacity |

The different tuning budgets reflect computational considerations and the use of global-search findings to guide capacity-specific optimization.

Because tuning budgets and search spaces differ, the results compare the implemented strategies rather than isolating the effect of the optimization objectives completely.

### Stage 4 – Final Evaluation

Models are evaluated using:

- PR-AUC
- ROC-AUC
- Precision@K
- Recall@K
- Lift@K

### Experimental Configuration

| Parameter | Configuration |
|---|---|
| Training split | 80% |
| Locked test split | 20% |
| Locked test observations | 50,736 |
| Cross-validation | 5-fold Stratified K-Fold |
| Random seed | 42 |
| Hyperparameter optimizer | Optuna |
| Selected model family | XGBoost |

The locked test set is reserved for final evaluation after model development and optimization.

---

## 5. Main Experimental Results

### 5.1 Global XGBoost Performance

The globally optimized XGBoost model achieved:

| Metric | Result |
|---|---|
| Mean cross-validation PR-AUC | 0.4389 |
| Pooled out-of-fold PR-AUC | 0.4384 |
| Pooled out-of-fold ROC-AUC | 0.8316 |
| Locked-test PR-AUC | 0.4258 |
| Locked-test ROC-AUC | 0.8280 |

Cross-validation and locked-test results refer to different evaluation settings.

### 5.2 Final Locked-Test Performance

| Capacity | Global P@K | Capacity-Specific P@K | Global Recall@K | Global Lift@K |
|---|---:|---:|---:|---:|
| 0.5% | 0.6890 | 0.6732 | 0.0248 | 4.94x |
| 1% | 0.6614 | 0.6614 | 0.0475 | 4.75x |
| 2.5% | 0.6052 | 0.6028 | 0.1086 | 4.34x |
| 5% | 0.5416 | 0.5388 | 0.1944 | 3.89x |
| 10% | 0.4773 | 0.4781 | 0.3426 | 3.43x |
| 20% | 0.3966 | 0.3963 | 0.5694 | 2.85x |

**Main observations:**

- Global optimization performs better at four capacity levels.
- Both strategies achieve identical Precision@K at 1%.
- Capacity-specific optimization provides only a marginal improvement at 10%.
- Precision@K generally decreases as screening capacity increases.
- Recall@K increases as a larger proportion of patients is selected.
- Lift@K remains above 2.8 across all evaluated capacities.

### 5.3 Comparison with Highest-k Loss

The results are additionally compared against published Highest-k Loss reference values.

| Capacity | Highest-k Loss | Global XGBoost | Capacity-Specific XGBoost |
|---|---:|---:|---:|
| 1% | 0.657 | 0.6614 | 0.6614 |
| 5% | 0.539 | 0.5416 | 0.5388 |
| 10% | 0.472 | 0.4773 | 0.4781 |

This is an **external benchmark comparison**, not a direct head-to-head evaluation, because experimental splits and training procedures differ.

---

## 6. Repository Structure

```text
Data_Mining-Group9-Diabetes-Risk-Screening-Optimization/
│
├── README.md
│
├── Data_Mining_Group_9_Report_Development_Process.ipynb
│
├── Group 9 + Optimizing Diabetes Risk Screening
│   for Capacity-Constrained Healthcare Systems
│   Using Boosting Models.pdf
│
├── diabetes_binary_health_indicators_BRFSS2015.csv
│
├── model_optimization_shortcut_files/
│   ├── global_xgboost.pkl
│   ├── xgboost_capacity_0.005.pkl
│   ├── xgboost_capacity_0.010.pkl
│   ├── xgboost_capacity_0.025.pkl
│   ├── xgboost_capacity_0.050.pkl
│   ├── xgboost_capacity_0.100.pkl
│   ├── xgboost_capacity_0.200.pkl
│   ├── global_optuna_study.pkl
│   ├── capacity_optuna_studies.pkl
│   ├── model_metadata.json
│   └── xgboost_best_params.json
│
└── deployment_section/
    ├── README.md
    ├── app/
    │   ├── app.py
    │   ├── batch_processing.py
    │   ├── logger.py
    │   ├── requirements.txt
    │   └── deployment_bundle/
    │       ├── diabetes_global_xgboost.ubj
    │       ├── feature_schema.json
    │       ├── model_metadata.json
    │       ├── capacity_config.json
    │       ├── shap_background.csv
    │       └── ...
    │
    └── development/
        ├── retrain_pipeline.py
        ├── Retrain_BRFSS2025_Group9.ipynb
        ├── requirements-retrain.txt
        ├── smoke_test.py
        ├── test_logger_offline.py
        ├── test_retrain_offline.py
        ├── report.md
        ├── report.json
        ├── RETRAINING_VI.md
        └── LOGGING_SETUP_VI.md
```

The repository separates the research workflow, saved optimization artifacts, deployed application, and offline development pipeline.

---

## 7. Reproducing the Research

The main research notebook is:

[Data_Mining_Group_9_Report_Development_Process.ipynb](Data_Mining_Group_9_Report_Development_Process.ipynb)

It can be opened in Google Colab or another compatible Jupyter environment.

### Option A – Full Optimization

This option reproduces the complete research workflow, including model training and Optuna optimization.

**Steps:**

1. Open the research notebook.
2. Execute environment setup and dependency installation.
3. Run data acquisition and validation.
4. Run exploratory data analysis.
5. Execute baseline model training and model screening.
6. Run global and capacity-specific Optuna optimization.
7. Execute final locked-test evaluation.
8. Generate model interpretation and calibration results.

Full optimization involves multiple cross-validation and hyperparameter search procedures, so it can be computationally expensive.

### Option B – Skip Optimization (Recommended)

This option allows reviewers to reproduce the evaluation workflow using saved research models without rerunning expensive optimization procedures.

**Step 1:** Open the research notebook and run environment setup, data acquisition, EDA, and data preparation.

**Step 2:** Skip the full Optuna optimization section.

**Step 3:** Download the saved models and optimization results from:

[Model Optimization Shortcut Files](model_optimization_shortcut_files/)

**Step 4:** Upload the required `.pkl` and `.json` files into the notebook working directory.

For Google Colab, this is typically `/content/`.

**Step 5:** Execute the **Model Optimization Shortcuts** loading cell.

**Step 6:** Continue with final evaluation, feature importance, SHAP interpretation, and calibration.

The shortcut files include:

- One globally optimized XGBoost model.
- Six capacity-specific XGBoost models.
- Global and capacity-specific Optuna study results.
- Hyperparameter configurations and model metadata.

> **Security note:** Pickle files should only be loaded from trusted sources. Python pickle deserialization can execute arbitrary code.

---

## 8. Model Interpretability

The study incorporates XGBoost feature importance and SHAP explanations.

### Global Interpretation

Global feature importance identifies the predictors most influential in the trained model.

Important health-related variables include:

- High blood pressure
- General health
- High cholesterol
- BMI
- Age
- Mobility difficulties

### Individual Interpretation

SHAP waterfall plots explain how individual feature values contribute to a selected observation's predicted model score.

These explanations support model transparency, but they describe predictive contributions rather than causal effects.

---

## 9. Streamlit Deployment

The project includes a Streamlit application for capacity-constrained diabetes screening demonstrations.

**Application directory:**

[deployment_section/app/](deployment_section/app/)

The deployed application uses the globally optimized XGBoost model, version **v1.0.0**.

### Main Features

| Feature | Description |
|---|---|
| Individual screening | Generates a screening score from 21 BRFSS indicators |
| SHAP explanation | Displays the three leading individual contributors |
| Risk context | Provides approximate percentile-based interpretation |
| Batch screening | Supports CSV and Excel upload |
| Patient ranking | Sorts observations by predicted score |
| Capacity selection | Selects the Top-K records according to available capacity |
| Export | Produces ranked results and selected records |
| Model monitoring context | Displays historical evaluation and calibration evidence |
| Prediction logging | Supports optional Google Sheets logging for demo predictions |

### Running the Application Locally

**1. Clone the repository**

```bash
git clone https://github.com/hung0302/Data_Mining-Group9-Diabetes-Risk-Screening-Optimization.git

cd Data_Mining-Group9-Diabetes-Risk-Screening-Optimization
```

**2. Create a virtual environment (recommended)**

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

**3. Install dependencies**

```bash
python -m pip install -r deployment_section/app/requirements.txt
```

**4. Run the application**

```bash
python -m streamlit run deployment_section/app/app.py
```

**5. Open the application**

Navigate to the local URL shown in the terminal, typically:

http://localhost:8501

Python 3.13 is preferred for compatibility with the recorded deployment environment.

### Capacity-Based Screening Example

Suppose a healthcare provider has:

- 1,000 candidate records.
- Capacity for 75 follow-up assessments.

The application ranks all 1,000 records using the global XGBoost model and selects the top 75.

The effective follow-up capacity is therefore **7.5%**.

The same trained model can be used for different capacities without retraining.

---

## 10. Monitoring and Retraining

The project includes a separate offline retraining framework to investigate model updating with later-period data.

The implemented research experiment uses **BRFSS 2025**, rather than outcomes collected from Streamlit application users.

### Retraining Architecture

The process follows these stages:

1. Validate historical and new labeled data.
2. Check feature schema and target compatibility.
3. Preserve the original BRFSS 2015 locked test set.
4. Combine historical training data with eligible new observations.
5. Train a candidate model using the production hyperparameters.
6. Evaluate the candidate against the existing model.
7. Review model promotion criteria.
8. Require an explicit decision before replacing the production model.

### BRFSS 2025 Experiment

| Component | Observations |
|---|---:|
| Original BRFSS 2025 respondents | 356,158 |
| Complete, compatible records | 201,093 |
| New training observations | 160,934 |
| New temporal holdout | 40,159 |
| Original 2015 training observations | 202,944 |
| Original 2015 locked test | 50,736 |

### Temporal Holdout Results

| Metric | Production v1.0.0 | Candidate v1.1.0 | Change |
|---|---:|---:|---:|
| PR-AUC | 0.4155 | 0.4197 | +0.0042 |
| ROC-AUC | 0.8085 | 0.8113 | +0.0028 |
| Brier score | 0.1085 | 0.1080 | -0.0005 |

The candidate produced modest performance improvements within the selected BRFSS 2025 holdout.

However, the experiment should be interpreted cautiously because only 56.5% of the 2025 records satisfied completeness and feature-compatibility requirements.

Potential selection bias and differences in survey coding may limit generalizability.

### Important Deployment Distinction

**Production model:** Global XGBoost v1.0.0

**Research candidate:** XGBoost v1.1.0

The candidate is not automatically deployed.

The Streamlit application continues using the existing production model unless a separate model promotion process is explicitly performed.

For technical details, see:

- [Deployment Documentation](deployment_section/README.md)
- [Retraining Instructions](deployment_section/development/RETRAINING_VI.md)
- [Retraining Evaluation Report](deployment_section/development/report.md)

---

## 11. Limitations and Responsible Use

Several limitations should be considered when interpreting the study.

**Survey-based data**

The BRFSS dataset relies on self-reported health and behavioral indicators, which may introduce reporting bias.

**Cross-sectional design**

The model predicts survey-reported diabetes status rather than future diabetes incidence. Predictive associations cannot establish causal relationships.

**Optimization comparison**

Global and capacity-specific optimization used different tuning budgets and search spaces. Therefore, observed performance differences cannot be attributed exclusively to the optimization objective.

**Population generalizability**

The original training data were collected in 2015. Performance on other populations and periods requires additional validation.

**Temporal validation**

The BRFSS 2025 analysis retained only records satisfying the required completeness and feature-compatibility criteria. This may introduce selection bias.

**Clinical validation**

The model has not undergone prospective clinical validation, and its predicted probabilities should not be interpreted as clinically calibrated diagnostic probabilities.

**Privacy and governance**

The application is designed for research and educational demonstrations. Real patient information should not be entered into unapproved systems or logging services.

> The screening framework is intended to support research into resource-aware prioritization, not replace medical professionals or clinical diagnostic procedures.

---

## 12. Team Members

**National Economics University – Data Mining, Group 9**

| Member | Student ID |
|---|---|
| Hoang Viet Hung | 11230749 |
| Le Phuong Khanh | 11230750 |
| Ninh Thi Minh Thuy | 11230779 |
| Chu Viet Anh | 11230730 |

**Instructor:** Nguyen Trong Nghia

---

## 13. Project Resources

| Resource | Link |
|---|---|
| Research notebook | [Open Notebook](Data_Mining_Group_9_Report_Development_Process.ipynb) |
| Final research report | [View PDF](Group%209%20%2B%20Optimizing%20Diabetes%20Risk%20Screening%20for%20Capacity-Constrained%20Healthcare%20Systems%20Using%20Boosting%20Models.pdf) |
| Dataset | [BRFSS 2015 CSV](diabetes_binary_health_indicators_BRFSS2015.csv) |
| Saved models and Optuna results | [Model Shortcuts](model_optimization_shortcut_files/) |
| Streamlit application | [Application Source](deployment_section/app/) |
| Deployment documentation | [Deployment README](deployment_section/README.md) |
| Retraining pipeline | [Retraining Source](deployment_section/development/retrain_pipeline.py) |
| Temporal evaluation | [Retraining Report](deployment_section/development/report.md) |

### External Documentation

- [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Optuna Documentation](https://optuna.readthedocs.io/)

For detailed academic references, methodology, experimental discussions, and comparisons with published work, consult the final research report.

---

**Data Mining – Group 9 | National Economics University | 2026**

*Developed for academic research, reproducible machine-learning experimentation, and educational demonstration.*
