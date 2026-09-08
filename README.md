# Loan Default Prediction

An end-to-end machine learning application that estimates the probability of loan default from borrower and loan attributes. The project includes a browser-based Flask app, an optional Streamlit interface, and a reproducible XGBoost training pipeline.

> **Important:** This project is for education and demonstration. It must not be used as the sole basis for real lending, credit, employment, housing, or other high-impact decisions.

## Overview

The application accepts a small set of loan features, preprocesses numeric and categorical values, and returns:

- An estimated default probability
- A decision label: `Accept`, `Waiting for review`, or `Reject`
- A risk level: `Low`, `Medium`, or `High`

The training pipeline uses an XGBoost classifier with one-hot encoding for categorical values, median/mode imputation, and class-imbalance handling.

## Features

- Flask web interface with form validation
- Optional Streamlit interface
- Consistent preprocessing between training and prediction
- Derived `loan_to_income` feature
- ROC-AUC evaluation during training
- Saved scikit-learn pipeline for repeatable predictions

## Input Features

| Feature      | Description                       |
| ------------ | --------------------------------- |
| `loan_amnt`  | Requested loan amount             |
| `annual_inc` | Borrower's annual income          |
| `int_rate`   | Interest rate as a percentage     |
| `term`       | Loan term in months: `36` or `60` |
| `grade`      | Loan grade from `A` through `G`   |
| `dti`        | Debt-to-income ratio percentage   |

## Project Structure

```text
.
├── app.py                 # Flask application entry point
├── application.py         # Optional Streamlit application
├── main.py                # Train and evaluate the model
├── predict.py             # Load the pipeline and make predictions
├── loan_pipeline.pkl      # Trained pipeline used by the apps
├── requirements.txt       # Python dependencies
├── templates/index.html   # Flask HTML template
├── static/style.css       # Flask styles
├── results/               # Evaluation plots
├── docs/                  # Reports and presentation material
└── .gitignore             # Local data and generated-file rules
```

## Requirements

- Python 3.10 or newer
- `pip`
- The trained `loan_pipeline.pkl` file for prediction
- The original `loan.csv` dataset only when retraining

## Installation

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install the project packages:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the Flask Application

The Flask app loads the included `loan_pipeline.pkl` model.

```bash
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

## Run the Streamlit Application

```bash
streamlit run application.py
```

## Train the Model

Training expects a source dataset named `loan.csv` in the project root. The dataset is excluded from Git because it is approximately 1.2 GB and is too large for a normal GitHub repository.

After obtaining the dataset through an approved source and placing it locally, run:

```bash
python main.py
```

The script cleans the data, creates the target label, trains the pipeline, prints the ROC-AUC score, and writes a new `loan_pipeline.pkl` file.

## Model Workflow

1. Remove sparse columns and records without a usable target label.
2. Convert `term`, `int_rate`, and `dti` into model-ready values.
3. Remove identifiers and post-loan leakage columns.
4. Create the `loan_to_income` feature.
5. Split the data into stratified training and test sets.
6. Impute missing values and one-hot encode categorical features.
7. Train an XGBoost classifier and evaluate it with ROC-AUC.

## Data and Privacy

The raw dataset, local archives, Python caches, and auxiliary generated model files are ignored by Git. Do not commit personal data, credentials, environment files, or any dataset that you do not have permission to redistribute.

## Limitations

- Predictions reflect the data and assumptions used during training.
- The displayed thresholds are application rules, not validated lending policy.
- Model performance should be evaluated for calibration, fairness, drift, and subgroup impact before any real-world use.
- This repository does not provide production authentication, monitoring, or deployment configuration.

## License

No license has been selected for this project yet. Add a `LICENSE` file before distributing or accepting external contributions.
