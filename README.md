
# Optimizing Diabetes Risk Screening for Capacity-Constrained Healthcare Systems Using Boosting Models

**National Economics University (NEU)**  
College of Technology · Faculty of Data Science and Artificial Intelligence

**Data Mining | Group 9 | November 2026**

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![Streamlit](https://img.shields.io/badge/Deployment-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Status](https://img.shields.io/badge/Status-Live%20Research%20Demo-brightgreen)

> **Research disclaimer:** This project is an academic research prototype for survey-based diabetes-status screening and resource-constrained patient prioritization. It is not a clinically validated diagnostic system and must not be used to make medical or treatment decisions.

---

## Quick Access

| Resource | Link |
|---|---|
| **Live Streamlit Application** | [Launch Diabetes Screening App](https://group9-diabetes-screening-hungthuyvanhkhanh.streamlit.app/) |
| **Final Research Report** | [Read the PDF](Group%209%20%2B%20Optimizing%20Diabetes%20Risk%20Screening%20for%20Capacity-Constrained%20Healthcare%20Systems%20Using%20Boosting%20Models.pdf) |
| **Research Notebook** | [Explore the Notebook](Data_Mining_Group_9_Report_Development_Process.ipynb) |
| **Saved Models & Optuna Studies** | [Optimization Shortcut Files](model_optimization_shortcut_files/) |
| **Deployment Documentation** | [Deployment README](deployment_section/README.md) |
| **Data Mining Course Website** | [Course Materials](https://nghianguyen7171.github.io/Data_mining/) |

---

## 1. Project Overview

### 1.1 Background and Motivation

Machine learning has become increasingly relevant to diabetes risk prediction because it can integrate health conditions, behavioral indicators, demographic characteristics, and socioeconomic factors into predictive models.

However, conventional evaluation metrics such as accuracy, ROC-AUC, and PR-AUC do not always directly reflect the operational constraints faced by healthcare providers.

In practice, healthcare systems may have sufficient resources to follow up only a limited proportion of individuals. Under these conditions, the central challenge is not simply to classify patients correctly, but to **prioritize the highest-risk individuals within a fixed screening capacity**.

This motivates a shift from conventional classification toward **capacity-constrained patient prioritization**, where model performance is evaluated within the Top-K proportion of individuals ranked by predicted risk.

### 1.2 Research Objective

This study develops and evaluates an XGBoost-based diabetes screening framework that supports patient prioritization under varying healthcare capacity constraints.

The central investigation compares two optimization strategies:

**Global Optimization**

A single XGBoost model is optimized to maximize cross-validation PR-AUC and subsequently evaluated across multiple healthcare capacity levels.

**Capacity-Specific Optimization**

Six XGBoost models are optimized separately to maximize Precision@K at their corresponding capacity levels.

The study investigates whether optimizing separate models for specific screening capacities produces meaningful improvements over maintaining one globally optimized model.

### 1.3 Main Contributions

This project makes four contributions:

**1. Capacity-aware model evaluation**

The study evaluates patient prioritization using Precision@K, Recall@K, and Lift@K across six capacity levels rather than relying exclusively on global classification metrics.

**2. Comparison of optimization strategies**

A globally optimized XGBoost model is compared with six capacity-specific XGBoost models to investigate whether specialized optimization produces meaningful out-of-sample improvements.

**3. External benchmark comparison**

The study compares its results with published Highest-k Loss findings at the 1%, 5%, and 10% capacity levels, while recognizing differences in experimental procedures.

**4. Interpretable deployment framework**

The selected model is incorporated into a Streamlit research prototype supporting individual screening scores, SHAP explanations, batch ranking, capacity-based selection, and an experimental monitoring and retraining workflow.

### 1.4 Key Finding

> **Capacity-specific optimization did not consistently outperform global optimization on the locked test set.**

The global XGBoost model performed better at four of six capacity levels, tied at one level, and was only marginally worse at another.

The results suggest that a single globally optimized model can provide flexible, capacity-aware prioritization without the operational complexity of maintaining six separately optimized models.

---

## 2. Research Questions

### Primary Research Question

How effectively can survey-based health and behavioral indicators be used to predict diabetes risk under different healthcare capacity constraints?

The study addresses four research questions:

**RQ1 — Associated Factors**

Which health, behavioral, demographic, and socioeconomic factors are most strongly associated with diabetes risk?

**RQ2 — Capacity-Constrained Prediction**

How effectively can machine-learning models identify high-risk individuals across different healthcare capacity levels?

**RQ3 — Optimization Strategy**

Does capacity-specific optimization improve Precision@K compared with a globally optimized model?

**RQ4 — Practical Deployment**

How can the resulting model be translated into an interpretable and deployable screening workflow for capacity-constrained healthcare settings?

---

## 3. Dataset

### 3.1 Data Source

The research uses the **CDC Diabetes Health Indicators dataset**, derived from the 2015 Behavioral Risk Factor Surveillance System (BRFSS).

- **Official source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators)
- **Repository dataset:** [BRFSS 2015 CSV](diabetes_binary_health_indicators_BRFSS2015.csv)

### 3.2 Why BRFSS 2015?

BRFSS 2015 was selected as the primary development cohort for three methodological reasons.

**Benchmark comparability**

The dataset provides a common historical data context for comparison with the published Highest-k Loss study, which also examined the BRFSS-derived diabetes dataset.

Although the studies use different experimental splits and training procedures, selecting the same underlying cohort facilitates a more meaningful external benchmark comparison than using an entirely different survey year.

**Experimental consistency**

The primary research objective is to compare global and capacity-specific optimization. Conducting these experiments within a single historical cohort helps avoid introducing additional temporal shifts, questionnaire changes, or population differences into the comparison.

**Temporal evaluation**

Retaining BRFSS 2015 as the development cohort also enables a separate evaluation using more recent BRFSS data.

This allows the study to investigate whether a model trained on historical data remains effective when evaluated against a later-period population and whether offline retraining offers measurable improvements.

Accordingly, **BRFSS 2015 is used for primary model development and comparison**, while BRFSS 2025 is used for the experimental temporal validation and retraining analysis.

### 3.3 Dataset Characteristics

| Property | Description |
|---|---|
| Source | CDC BRFSS 2015 |
| Observations | 253,680 |
| Predictors | 21 |
| Target variable | `Diabetes_binary` |
| Problem type | Binary classification |
| Positive observations | 35,346 |
| Negative observations | 218,334 |
| Positive prevalence | 13.93% |
| Missing values | None detected |

The dataset contains variables representing health conditions, behavioral characteristics, demographic information, socioeconomic status, and access to healthcare.

**Continuous indicators**

- BMI
- MentHlth
- PhysHlth

**Ordinal indicators**

- GenHlth
- Age
- Education
- Income

The remaining predictors are primarily binary indicators.

### 3.4 Data Preparation

The data preparation workflow includes:

1. Data acquisition and basic validation.
2. Missing-value and invalid-range checks.
3. Duplicate and outlier assessment.
4. Exploratory data analysis.
5. Feature and target separation.
6. Stratified training and test splitting.
7. Model-specific preprocessing.

No missing values or invalid-range observations were detected in the original dataset.

Identical records were retained because no respondent identifier was available to establish whether they represented erroneous duplicates.

Potential IQR-based outliers were also retained when their values fell within valid survey ranges.

Standardization was applied to Logistic Regression, while boosting models used the original feature scales.

---

## 4. Research Methodology

### 4.1 Research Workflow

The study follows five main stages:

```text
BRFSS 2015 Dataset
         |
         v
Data Exploration and Preparation
         |
         v
Baseline and Boosting Model Screening
         |
         v
Global vs. Capacity-Specific Optimization
         |
         v
Locked-Test Evaluation and Interpretation
         |
         v
Streamlit Research Prototype
         |
         v
Offline Temporal Evaluation and Retraining
```

### 4.2 Baseline and Model Screening

Logistic Regression was used as the baseline model.

Four boosting algorithms were then evaluated:

- HistGradientBoosting
- XGBoost
- LightGBM
- CatBoost

#### Default Model Screening Results

| Model | PR-AUC | ROC-AUC |
|---|---:|---:|
| Logistic Regression | 0.4073 | 0.8228 |
| XGBoost | 0.4225 | 0.8260 |
| CatBoost | 0.4254 | 0.8281 |
| HistGradientBoosting | 0.4343 | 0.8301 |
| LightGBM | **0.4346** | **0.8303** |

All four boosting models outperformed the Logistic Regression baseline in PR-AUC.

#### Why LightGBM and XGBoost?

LightGBM and XGBoost were retained for comparative tuning based on both empirical screening results and the research objectives.

**LightGBM** achieved the highest default PR-AUC and ROC-AUC among the screened boosting algorithms, making it a strong empirical candidate.

**XGBoost** was retained as a literature-supported candidate because previous diabetes prediction studies had reported competitive results using this algorithm.

The research aimed to investigate whether an established boosting approach could be further optimized for capacity-constrained patient prioritization.

Although HistGradientBoosting achieved a higher default PR-AUC than XGBoost, the decision to retain XGBoost was not based solely on initial model ranking. It also reflected the study's literature-informed modeling direction.

Importantly, this selection does not establish that XGBoost is universally superior to HistGradientBoosting.

#### Results After Light Tuning

| Model | 0.5% | 1% | 2.5% | 5% | 10% | 20% |
|---|---:|---:|---:|---:|---:|---:|
| LightGBM | 0.7054 | 0.6768 | **0.6271** | 0.5609 | 0.4902 | 0.4000 |
| XGBoost | **0.7084** | **0.6857** | 0.6248 | **0.5617** | **0.4916** | **0.4007** |

After light tuning:

- XGBoost achieved PR-AUC = **0.4372**.
- LightGBM achieved PR-AUC = **0.4362**.
- XGBoost achieved higher Precision@K at five of six evaluated capacity levels.

XGBoost was therefore selected for deeper global and capacity-specific optimization.

### 4.3 Experimental Configuration

| Component | Configuration |
|---|---|
| Train-test split | 80% / 20% |
| Locked test observations | 50,736 |
| Cross-validation | 5-fold Stratified K-Fold |
| Random seed | 42 |
| Selected algorithm | XGBoost |
| Hyperparameter optimizer | Optuna |
| Global optimization objective | Mean CV PR-AUC |
| Capacity-specific objective | Mean CV Precision@K |
| Global tuning budget | 30-trial pilot + 100-trial full search |
| Capacity-specific tuning budget | 30 trials per capacity |

The same data split, cross-validation framework, random seed, and model family were maintained for the global and capacity-specific experiments.

However, the tuning budgets and hyperparameter search spaces were not identical. Therefore, the comparison evaluates the implemented optimization strategies rather than isolating the effect of the optimization objectives alone.

### 4.4 Global Optimization

The global strategy optimizes a single XGBoost model using mean five-fold PR-AUC.

The initial Optuna pilot explored the boosting-iteration range from 200 to 1,000.

Because promising configurations occurred near the upper search boundary, the full optimization expanded the range to 200–1,600.

The selected global model was evaluated at every capacity level.

### 4.5 Capacity-Specific Optimization

Six separate XGBoost models were optimized for:

| Capacity | Approximate Locked-Test Selection Size |
|---|---:|
| 0.5% | 254 |
| 1% | 508 |
| 2.5% | 1,269 |
| 5% | 2,537 |
| 10% | 5,074 |
| 20% | 10,148 |

Each capacity-specific model was optimized using mean five-fold Precision@K at its corresponding capacity.

The models were trained on complete training folds. Capacity awareness was introduced through the hyperparameter optimization objective rather than by restricting training to the Top-K observations.

---

## 5. Evaluation Metrics

### 5.1 Precision@K

Precision@K measures the proportion of actual positive observations among the K highest-ranked individuals.

\[
\text{Precision@K} =
\frac{\text{True positives among Top-K}}{K}
\]

This is the primary operational metric for evaluating how effectively a model concentrates positive cases within limited screening capacity.

### 5.2 Recall@K

Recall@K measures the proportion of all actual positive cases captured within the selected Top-K observations.

\[
\text{Recall@K} =
\frac{\text{True positives among Top-K}}
{\text{Total positive observations}}
\]

### 5.3 Lift@K

Lift@K compares the positive rate among selected observations with the overall population prevalence.

\[
\text{Lift@K} =
\frac{\text{Precision@K}}
{\text{Population prevalence}}
\]

A Lift@K greater than 1 indicates that the selected subgroup contains a higher concentration of positive cases than random selection would be expected to achieve.

### 5.4 Global Discrimination Metrics

**PR-AUC** measures performance across the precision-recall curve and is particularly relevant when the positive class is less prevalent.

**ROC-AUC** measures the model's ability to discriminate between positive and negative observations across classification thresholds.

### 5.5 Brier Score and Calibration

The Brier score evaluates probabilistic prediction error:

\[
\text{Brier Score} =
\frac{1}{N}\sum_{i=1}^{N}(p_i-y_i)^2
\]

Lower values indicate lower mean squared prediction error.

Calibration analysis provides additional information about the agreement between predicted probabilities and observed positive rates.

However, a Brier score alone does not establish good calibration.

In this study, calibration is treated as a supplementary evaluation because the primary objective is capacity-based ranking and prioritization.

No probability recalibration was performed on the deployed model.

---

## 6. Main Experimental Results

### 6.1 Global XGBoost Performance

| Metric | Result |
|---|---:|
| Mean cross-validation PR-AUC | 0.4389 |
| Pooled out-of-fold PR-AUC | 0.4384 |
| Pooled out-of-fold ROC-AUC | 0.8316 |
| Locked-test PR-AUC | **0.4258** |
| Locked-test ROC-AUC | **0.8280** |
| Locked-test Brier score | **0.0987** |

The locked-test discrimination metrics were slightly lower than the corresponding out-of-fold metrics.

The results provide evidence of performance on the held-out sample while highlighting the importance of evaluating models beyond cross-validation.

The Brier score provides supplementary probabilistic evaluation; it should not be interpreted as proof of clinical calibration.

### 6.2 Final Locked-Test Capacity Performance

All finalized models were evaluated on the locked test set containing **50,736 observations**.

| Capacity | Global P@K | Capacity-Specific P@K | Global Recall@K | Global Lift@K |
|---|---:|---:|---:|---:|
| 0.5% | **0.6890** | 0.6732 | 0.0248 | 4.94x |
| 1% | 0.6614 | 0.6614 | 0.0475 | 4.75x |
| 2.5% | **0.6052** | 0.6028 | 0.1086 | 4.34x |
| 5% | **0.5416** | 0.5388 | 0.1944 | 3.89x |
| 10% | 0.4773 | **0.4781** | 0.3426 | 3.43x |
| 20% | **0.3966** | 0.3963 | 0.5694 | 2.85x |

#### Main Findings

**Global-model robustness**

Global optimization achieved higher Precision@K at four capacity levels and identical performance at 1%.

**Limited benefits from specialization**

Capacity-specific optimization performed marginally better at 10%, but did not consistently improve locked-test performance.

**Precision-recall trade-off**

Precision@K generally decreased as capacity expanded, while Recall@K increased.

**Effective risk concentration**

Lift@K remained above 2.8 across all evaluated capacities, indicating a substantially higher concentration of positive cases in the selected groups relative to the population prevalence.

### 6.3 Comparison with Published Highest-k Loss Results

| Capacity | Highest-k Loss P@K | Global XGBoost P@K | Capacity-Specific P@K |
|---|---:|---:|---:|
| 1% | 0.657 | **0.6614** | 0.6614 |
| 5% | 0.539 | **0.5416** | 0.5388 |
| 10% | 0.472 | 0.4773 | **0.4781** |

The global XGBoost model achieved slightly higher reported values than the published Highest-k Loss reference at all three comparable capacities.

**Important:** These results represent an external literature benchmark comparison, not a direct head-to-head experiment.

Differences in training procedures, data splits, and experimental conditions prevent a definitive conclusion about algorithmic superiority.

### 6.4 Overall Research Conclusion

The experimental evidence supports selecting the global XGBoost model for the proposed research prototype.

It offers competitive capacity-based ranking while avoiding the additional training, validation, deployment, and maintenance requirements of six separate models.

---

## 7. Model Interpretability

The study uses feature importance and SHAP to support model transparency.

### 7.1 Global Feature Importance

The selected global XGBoost model identifies several influential health-related predictors, including:

- High blood pressure (HighBP)
- General health (GenHlth)
- High cholesterol (HighChol)
- Body mass index (BMI)
- Age
- Mobility difficulties (DiffWalk)

These variables contribute to the model's predictive behavior but should not be interpreted as causal determinants of diabetes.

### 7.2 Individual SHAP Explanations

SHAP waterfall plots explain how individual feature values contribute to a selected observation's model prediction.

The research report presents an example observation with a predicted diabetes-status probability of approximately 0.189.

The example illustrates how different health indicators can increase or decrease an individual's predicted score.

SHAP contributions are interpreted on the model's output scale, including log-odds where applicable.

---

## 8. Repository Structure

```text
Data_Mining-Group9-Diabetes-Risk-Screening-Optimization/
│
├── README.md
│
├── Data_Mining_Group_9_Report_Development_Process.ipynb
│
├── Group 9 + Optimizing Diabetes Risk Screening for Capacity-Constrained Healthcare Systems Using Boosting Models.pdf
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
    ├── .gitignore
    │
    ├── app/
    │   ├── app.py
    │   ├── batch_processing.py
    │   ├── logger.py
    │   ├── requirements.txt
    │   └── deployment_bundle/
    │       ├── diabetes_global_xgboost.ubj
    │       ├── portable_model_checksum.txt
    │       ├── feature_schema.json
    │       ├── model_metadata.json
    │       ├── capacity_config.json
    │       ├── training_params.json
    │       ├── risk_reference.json
    │       ├── shap_background.csv
    │       ├── calibration_report.json
    │       ├── calibration_table.csv
    │       ├── calibration_curve.png
    │       ├── sample_input.csv
    │       ├── sample_output.json
    │       ├── batch_example_20.xlsx
    │       ├── prediction_logging_schema.csv
    │       └── retraining_config.json
    │
    └── development/
        ├── retrain_pipeline.py
        ├── Retrain_BRFSS2025_Group9.ipynb
        ├── requirements-retrain.txt
        ├── smoke_test.py
        ├── test_logger_offline.py
        ├── test_retrain_offline.py
        ├── Code.gs
        ├── LOGGING_SETUP_VI.md
        ├── RETRAINING_VI.md
        ├── retraining_verification_v1.0.0.md
        ├── report.md
        ├── report.json
        ├── google_sheet_headers.csv
        ├── calibration_curve_v1.1.0.png
        └── make_demo_prediction_log.py
```

The repository separates:

- **Research and experimentation:** Main notebook and final report.
- **Saved optimization artifacts:** Pretrained models and Optuna results.
- **Application runtime:** Streamlit code and production model bundle.
- **Development workflows:** Offline retraining, testing, and experimental evaluation.

---

## 9. Reproducing the Research

### 9.1 Main Research Notebook

**Notebook:** [Data_Mining_Group_9_Report_Development_Process.ipynb](Data_Mining_Group_9_Report_Development_Process.ipynb)

The notebook can be opened in Google Colab or another compatible Jupyter environment.

Two execution approaches are provided.

### Option A — Full Optimization

This approach runs the complete research workflow, including computationally expensive model training and hyperparameter optimization.

**Steps**

1. Open the research notebook.
2. Execute environment setup and dependency installation.
3. Run data acquisition and validation.
4. Execute exploratory data analysis.
5. Run baseline training and boosting model screening.
6. Execute global Optuna optimization.
7. Execute capacity-specific Optuna optimization.
8. Run final evaluation on the locked test set.
9. Generate interpretability and calibration outputs.

The full optimization workflow includes multiple cross-validation runs and may require substantial computational resources.

### Option B — Model Optimization Shortcut (Recommended)

This approach uses saved research artifacts to reproduce downstream evaluation without rerunning the full hyperparameter search.

**Step 1: Open the notebook**

Open the main `.ipynb` file in Google Colab.

**Step 2: Run preliminary sections**

Execute:

- Environment setup.
- Data acquisition and validation.
- Exploratory data analysis.
- Data preparation.

**Step 3: Skip full Optuna optimization**

Navigate past the computationally expensive global and capacity-specific optimization sections.

**Step 4: Download saved artifacts**

Download the files from:

[model_optimization_shortcut_files/](model_optimization_shortcut_files/)

The directory contains the saved global model, six capacity-specific models, Optuna studies, and model metadata.

**Step 5: Upload model files**

Upload the required `.pkl` and `.json` files into the notebook working directory.

For Google Colab, the default working directory is typically `/content/`.

**Step 6: Execute the loading cell**

Navigate to the **Model Optimization Shortcuts** section and execute the artifact-loading cell.

**Step 7: Continue evaluation**

Run the remaining sections covering:

- Final locked-test evaluation.
- Model comparison.
- Feature importance.
- SHAP explanations.
- Calibration analysis.

> **Security:** Pickle files can execute code during deserialization. Only load saved model artifacts from sources you trust.

### 9.2 Dataset Acquisition Fallback

The notebook first attempts to retrieve the dataset from its original source.

If that acquisition fails, the data-loading workflow can use the repository's backup CSV.

Backup dataset:

[diabetes_binary_health_indicators_BRFSS2015.csv](diabetes_binary_health_indicators_BRFSS2015.csv)

The dataset is validated against the expected structure of 253,680 observations and 22 columns, including the target.

---

## 10. Streamlit Deployment

### Live Application

**[Launch the Diabetes Screening Application](https://group9-diabetes-screening-hungthuyvanhkhanh.streamlit.app/)**

The project includes an interactive Streamlit research prototype hosted on Streamlit Community Cloud.

The application loads the selected global XGBoost model and supports individual screening and cohort-level patient prioritization.

**Currently deployed model:** Global XGBoost v1.0.0

The application does not train or retrain the model during inference.

### 10.1 Main Features

| Feature | Description |
|---|---|
| Single-patient screening | Generates a model-based screening score using 21 BRFSS predictors |
| SHAP interpretation | Displays leading individual feature contributions |
| Risk percentile context | Provides approximate score-percentile information |
| Batch screening | Accepts compatible CSV and Excel records |
| Patient ranking | Orders uploaded records by predicted score |
| Capacity-based selection | Selects the highest-ranked observations within a defined follow-up capacity |
| Result export | Exports ranked results and selected records |
| Calibration information | Presents stored locked-test calibration evidence |
| Prediction logging | Optionally records demo predictions through Google Apps Script and Google Sheets |

### 10.2 Running the Application Locally

**Step 1 — Clone the repository**

```bash
git clone https://github.com/hung0302/Data_Mining-Group9-Diabetes-Risk-Screening-Optimization.git

cd Data_Mining-Group9-Diabetes-Risk-Screening-Optimization
```

**Step 2 — Create a virtual environment**

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

**Step 3 — Install dependencies**

```bash
python -m pip install -r deployment_section/app/requirements.txt
```

**Step 4 — Launch Streamlit**

```bash
python -m streamlit run deployment_section/app/app.py
```

**Step 5 — Open the local application**

Use the local URL displayed in the terminal, usually:

http://localhost:8501

Python 3.13 is recommended for consistency with the recorded deployment environment.

### 10.3 Capacity-Based Selection

The application allows users to specify the number of records that can receive follow-up.

For a cohort containing N valid observations and an available capacity of C:

\[
m = \min(C,N)
\]

The effective selection capacity is:

\[
K = \frac{m}{N}\times100\%
\]

For example:

- Cohort size: 1,000 observations.
- Available follow-up capacity: 75 observations.
- Selected records: Top 75.
- Effective capacity: 7.5%.

The same global model can support different capacity constraints without requiring separate specialized models.

### 10.4 Deployment Architecture

```text
User Input
    |
    v
Streamlit Interface
    |
    v
Input Validation
    |
    v
Global XGBoost v1.0.0
    |
    +----> Individual Score
    |           |
    |           +----> SHAP Explanation
    |           |
    |           +----> Optional Demo Logging
    |
    +----> Batch Predictions
                |
                v
          Descending Ranking
                |
                v
          Capacity Selection
                |
                v
           Export Results
```

The application uses a versioned deployment bundle containing model artifacts, feature schema information, capacity metrics, and supplementary evaluation resources.

A checksum is used to verify the integrity of the portable XGBoost model file before loading.

For implementation details, see:

[Deployment Documentation](deployment_section/README.md)

---

## 11. Monitoring and Retraining

### 11.1 Framework Overview

The project includes a separate offline retraining pipeline designed to investigate how newer labeled data may affect model performance.

This pipeline is independent of the Streamlit application's inference process.

The implemented temporal research experiment uses BRFSS 2025 records rather than actual outcomes collected from application users.

### 11.2 BRFSS 2025 Experimental Setup

| Component | Observations |
|---|---:|
| Original BRFSS 2025 respondents | 356,158 |
| Complete and usable records | 201,093 |
| New training observations | 160,934 |
| New temporal holdout | 40,159 |
| Historical 2015 training observations | 202,944 |
| Original 2015 locked test | 50,736 |

The 2015 locked test set was preserved to evaluate whether the candidate model experienced performance regression on the original evaluation cohort.

The new temporal holdout was used to investigate performance on later-period observations.

### 11.3 Temporal Holdout Results

| Metric | Production v1.0.0 | Candidate v1.1.0 | Change |
|---|---:|---:|---:|
| PR-AUC | 0.4155 | 0.4197 | +0.0042 |
| ROC-AUC | 0.8085 | 0.8113 | +0.0028 |
| Brier score | 0.1085 | 0.1080 | -0.0005 |
| Precision@1% | 0.6294 | 0.6318 | +0.0025 |
| Precision@5% | 0.5378 | 0.5433 | +0.0055 |
| Precision@10% | 0.4771 | 0.4858 | +0.0087 |

The candidate model demonstrated modest improvements on the selected 2025 temporal holdout.

However, the results are not sufficient to establish improvement across all future populations or deployment environments.

### 11.4 Data Comparability

The 2025 dataset was harmonized to the 21-feature model schema and binary target definition used in the project.

Reported diabetes was assigned to the positive class, while other specified status categories were assigned to the negative class.

Despite this harmonization, potential differences in survey questionnaires, response coding, and population composition remain important limitations.

Only 201,093 of 356,158 records satisfied the required completeness and compatibility criteria.

This selection may introduce bias and limit the generalizability of the temporal evaluation.

### 11.5 Retraining Process

The offline pipeline implements the following workflow:

1. Load historical and newly labeled data.
2. Validate feature schema, values, and labels.
3. Check retraining eligibility.
4. Preserve the original locked test set.
5. Prepare new training and holdout records.
6. Train a candidate model.
7. Evaluate candidate and production models on the same evaluation samples.
8. Apply predefined performance and governance criteria.
9. Stage the candidate for review.
10. Promote only through an explicit authorized action.

### 11.6 Production vs. Research Candidate

| Component | Status |
|---|---|
| Global XGBoost v1.0.0 | Deployed production artifact |
| Retrained XGBoost v1.1.0 | Research candidate |
| Streamlit inference | Uses v1.0.0 |
| Offline retraining pipeline | Separate from application runtime |
| Automatic model replacement | Not enabled |

**The experimental candidate v1.1.0 has not been automatically promoted to production.**

The deployment framework intentionally separates model evaluation from model replacement.

### 11.7 Retraining Documentation

Additional technical resources:

- [Retraining Pipeline](deployment_section/development/retrain_pipeline.py)
- [Retraining Instructions](deployment_section/development/RETRAINING_VI.md)
- [Temporal Evaluation Report](deployment_section/development/report.md)
- [Retraining Notebook](deployment_section/development/Retrain_BRFSS2025_Group9.ipynb)

---

## 12. Limitations and Responsible Use

### 12.1 Survey-Based Labels

The research relies on self-reported health information, which may be affected by recall error, reporting bias, and inconsistencies in survey responses.

### 12.2 Cross-Sectional Prediction

The primary model predicts survey-reported diabetes status rather than future disease incidence.

Its predictions must not be interpreted as causal explanations or future clinical diagnoses.

### 12.3 Optimization Comparability

Global and capacity-specific optimization used different trial budgets and hyperparameter search spaces.

Therefore, the observed performance differences cannot be attributed exclusively to the choice of optimization objective.

### 12.4 External Benchmarking

The Highest-k Loss comparison uses published results rather than a direct reproduction under identical experimental conditions.

The comparisons should be interpreted as contextual evidence rather than definitive proof of superiority.

### 12.5 Temporal Generalizability

The primary development dataset originates from 2015.

Survey distributions, population characteristics, and data collection procedures may change over time.

The BRFSS 2025 analysis provides an additional evaluation context, but does not eliminate the need for broader external validation.

### 12.6 Probability Calibration

The study reports Brier score and calibration evidence as supplementary evaluation results.

The deployed probabilities have not undergone a separate recalibration procedure and should not be interpreted as clinically validated risk estimates.

### 12.7 Privacy and Clinical Safety

The Streamlit application is an academic demonstration.

It is not designed or certified for processing confidential patient information in a clinical environment.

Users should use synthetic, anonymous, or otherwise appropriate demonstration data.

Predictions should not be used independently to determine diagnosis, treatment, or medical eligibility.

---

## 13. Team Members

**National Economics University — Data Mining, Group 9**

| Member | Student ID | Main Contribution |
|---|---|---|
| Chu Viet Anh | 11230730 | Continuous learning framework, monitoring, retraining, and temporal evaluation |
| Hoang Viet Hung | 11230749 | Data analysis, model development, optimization, and experimental evaluation |
| Le Phuong Khanh | 11230750 | Streamlit deployment, interpretability, batch prioritization, and integration |
| Ninh Thi Minh Thuy | 11230779 | Research background, literature review, methodology, and report synthesis |

**Instructor:** Nguyen Trong Nghia

**Institution:** National Economics University, Hanoi, Vietnam

**Academic Year:** 2026

---

## 14. Project Resources

| Resource | Link |
|---|---|
| Final Research Report | [View PDF](Group%209%20%2B%20Optimizing%20Diabetes%20Risk%20Screening%20for%20Capacity-Constrained%20Healthcare%20Systems%20Using%20Boosting%20Models.pdf) |
| Main Research Notebook | [Open Notebook](Data_Mining_Group_9_Report_Development_Process.ipynb) |
| BRFSS 2015 Dataset | [Download CSV](diabetes_binary_health_indicators_BRFSS2015.csv) |
| Saved Optimization Models | [Model Files](model_optimization_shortcut_files/) |
| Live Streamlit Application | [Launch Application](https://group9-diabetes-screening-hungthuyvanhkhanh.streamlit.app/) |
| Streamlit Source Code | [Application Directory](deployment_section/app/) |
| Deployment Documentation | [Deployment README](deployment_section/README.md) |
| Offline Retraining Pipeline | [Python Source](deployment_section/development/retrain_pipeline.py) |
| Retraining Evaluation | [Research Report](deployment_section/development/report.md) |
| Logging Configuration | [Setup Guide](deployment_section/development/LOGGING_SETUP_VI.md) |
| Course Website | [Data Mining Course Materials](https://nghianguyen7171.github.io/Data_mining/) |

### External Documentation

- [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [scikit-learn Documentation](https://scikit-learn.org/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [CatBoost Documentation](https://catboost.ai/)
- [Optuna Documentation](https://optuna.readthedocs.io/)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [Streamlit Documentation](https://docs.streamlit.io/)

For the complete literature review, academic references, experimental methodology, and interpretation of results, refer to the final research report.

---

## 15. Final Remarks

This research examines whether capacity-specific optimization is necessary for effective diabetes-status screening when healthcare follow-up resources are limited.

The experimental results indicate that a single globally optimized XGBoost model can provide competitive Precision@K across multiple screening capacities without requiring a separate model for every operating threshold.

The Streamlit prototype demonstrates how this approach can be translated into an interpretable, capacity-aware research workflow.

Future work should investigate broader external validation, model calibration, population shift, fairness, and the requirements for responsible clinical evaluation.

---

**Data Mining – Group 9 | National Economics University | November 2026**

*Academic research · Reproducible experimentation · Interpretable machine learning · Capacity-aware screening*
