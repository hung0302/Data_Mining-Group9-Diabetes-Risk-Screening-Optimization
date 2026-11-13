# Diabetes Screening Deployment and Research Repository

## Overview

This repository contains the final submission for the Group 9 Data Mining project on diabetes screening risk prediction. The deployed application is a Streamlit-based prototype that uses the production global XGBoost model, version 1.0.0, to estimate screening priority from BRFSS-style survey responses. The repository also includes a separate development workflow for offline retraining and experimental validation with a candidate model version 1.1.0 built from later BRFSS data.

The currently deployed model remains the production artifact under the app bundle and is not replaced by the retraining candidate. The candidate model is preserved only for research evaluation and comparison; it is not promoted to runtime without an explicit manual deployment decision.

## Folder structure

```text
deployment_section/
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
│       ├── risk_reference.json
│       ├── capacity_config.json
│       ├── shap_background.csv
│       ├── sample_input.csv
│       ├── sample_output.json
│       ├── batch_example_20.xlsx
│       ├── calibration_report.json
│       ├── calibration_table.csv
│       ├── calibration_curve.png
│       ├── prediction_logging_schema.csv
│       ├── retraining_config.json
│       └── training_params.json
├── development/
│   ├── retrain_pipeline.py
│   ├── Retrain_BRFSS2025_Group9.ipynb
│   ├── requirements-retrain.txt
│   ├── smoke_test.py
│   ├── test_logger_offline.py
│   ├── test_retrain_offline.py
│   ├── Code.gs
│   ├── LOGGING_SETUP_VI.md
│   ├── RETRAINING_VI.md
│   ├── retraining_verification_v1.0.0.md
│   ├── google_sheet_headers.csv
│   ├── calibration_curve_v1.1.0.png
│   ├── report.json
│   ├── report.md
│   └── make_demo_prediction_log.py
├── README.md
└── .gitignore
```

The app bundle contains the exact production model and the locked-report artifacts used by the live Streamlit app. The development folder contains the research notebook, retraining pipeline, test scripts, and experimental reports.

## Streamlit application features

The deployed app in `app/app.py` provides:

- single-patient diabetes risk screening based on BRFSS input features;
- SHAP-based explanations for the three most influential features;
- risk percentile context derived from the supplied reference distribution;
- batch cohort upload in Excel or CSV format;
- ranking and Top-K selection using the global production model;
- locked-test calibration summaries and capacity metrics;
- optional Google Apps Script logging for classroom demo predictions.

The production model is the global XGBoost v1.0.0 model shipped in the app bundle. The app does not retrain or replace the model during runtime.

## Dependencies and how to run the app

From the repository root:

```bash
python -m pip install -r deployment_section/app/requirements.txt
python -m streamlit run deployment_section/app/app.py
```

If needed, run from a virtual environment and ensure that the app can access the resources in `deployment_section/app/deployment_bundle`.

## Development tests

The repository includes a focused offline smoke test and logging checks:

```bash
python deployment_section/development/smoke_test.py
python deployment_section/development/test_logger_offline.py
python deployment_section/development/test_retrain_offline.py
```

These checks validate the production bundle checksum, single-patient prediction output, SHAP shape, calibration data, and offline retraining logic without invoking a production promotion.

## Retraining pipeline organization

The experimental retraining pipeline lives in `deployment_section/development/retrain_pipeline.py`. It reads the production bundle from `deployment_section/app/deployment_bundle` and writes local retraining artifacts under `deployment_section/development/` such as:

- `data/` for the historical BRFSS dataset and any local new-record exports;
- `retraining_runs/` for candidate reports and evaluation summaries;
- `model_registry/` for staged or archived model outputs.

This separation ensures the deployed app and the research pipeline remain distinct. The script validates schema, evaluates candidate changes, and enforces promotion criteria, but it does not automatically replace the production model unless the user explicitly runs a promotion step.

## Reproducing the retraining experiment

To reproduce the research experiment, the following external data are required:

- BRFSS 2015 historical training data in the same 21-feature schema used by the app;
- a newer BRFSS data export with labels consistent with the same feature coding;
- local access to the project files in `deployment_section/development/` and the production bundle in `deployment_section/app/deployment_bundle/`.

The BRFSS 2015 file is included at the repository root (`diabetes_binary_health_indicators_BRFSS2015.csv`). The newer BRFSS 2025 cohort is not included; download it from the CDC and recode it with `development/Retrain_BRFSS2025_Group9.ipynb`. To run the pipeline from `development/`, point it to the root file:
`python retrain_pipeline.py --historical ../../diabetes_binary_health_indicators_BRFSS2015.csv --new-data <recoded_2025.csv> --ignore-schedule`

The general processing workflow is:

1. Prepare a local historical dataset and any new labeled cohort data.
2. Run the retraining script from the development folder.
3. Review the generated evaluation report in `development/retraining_runs/`.
4. Compare the candidate model to the production model v1.0.0 without changing the app bundle.

The current workspace includes the experimental result files and the notebook for reference, but the candidate v1.1.0 model is not the deployed production model.

## Limitations and disclaimer

This project is a research prototype and educational demonstration. It is intended for classroom analysis and reproducible evaluation rather than direct clinical deployment. The model-estimated probability and ranking output should not be interpreted as a validated medical diagnosis, and no patient-level prediction should be used as a treatment decision.

The app is intentionally designed to preserve the existing production model artifact and to keep the experimental retraining workflow separate. The candidate v1.1.0 is research-only and must not be described as deployed unless an explicit manual deployment step is performed.

The final production bundle remains the global XGBoost v1.0.0 model in `deployment_section/app/deployment_bundle/`. Any later retraining output is experimental and requires independent validation before any operational adoption.
