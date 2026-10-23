# JobGuard

**JobGuard is an explainable research prototype for screening suspicious job and internship postings.** It compares five classical NLP classifiers, surfaces language and rule-based signals, and provides safety guidance. It is a screening aid—not proof of fraud.

## Problem

Fake job ads can exploit job seekers through advance-fee requests, impersonated companies, suspicious links, or pressure to act quickly. JobGuard explores whether text classification can help people pause and verify an offer before sharing money or sensitive information.

## Dataset

- **Source:** EMSCAD / `fake_job_postings.csv`
- **Expected size:** 17,880 postings with a binary `fraudulent` label.
- **Class imbalance:** about 5% of postings are labeled fake.
- **Input fields:** `title`, `company_profile`, `description`, `requirements`, and `benefits` are joined into the model text.
- **Limitations:** EMSCAD is an older, predominantly English dataset. It may not represent present-day scams, regional language postings, or current job-market practices. Check dataset licensing and attribution requirements before redistribution.

## Method

1. Build a stratified 80/20 train/test split (`random_state=42`).
2. Split the training portion again for threshold selection.
3. Represent text using TF-IDF unigrams and bigrams, English stop-word removal, `min_df=3`, and sublinear term frequency.
4. Compare Naive Bayes, Logistic Regression, calibrated Linear SVM, SGD with log loss, and a size-limited Random Forest where the operating system allows scikit-learn's tree extension to load.
5. Run five-fold stratified cross-validation on the training-fit partition. Each fold fits its own TF-IDF vocabulary and IDF weights.
6. Select the maximum-F1 and high-recall operating points on a separate training validation split; retain the held-out test split for reporting.
7. Report accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrices, and ROC/PR curve points.
8. Save Logistic Regression coefficient terms, up to 15 test-set error examples, and a small set of labeled fake-posting TF-IDF vectors for cosine-similarity exploration.

The app's blended screening score is **75% selected-model fake probability + 25% severity-weighted rule score**. This heuristic blend is transparent but is **not itself a calibrated probability**. The gauge shows the selected model probability; the result label uses the combined score and user-adjustable threshold (default 0.50). The validation-derived model thresholds shown in Model Lab apply to the model probability by itself, not to this blended score.

The app defaults to Logistic Regression so coefficient-based explanations match the selected model; the measured best test F1 in this run was from calibrated Linear SVM. The user can select any trained model. The default always-real test accuracy was **95.162%**.

### Logistic Regression operating points (validation split)

| Operating point | Threshold | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Maximum validation F1 | 0.4104 | 0.869 | 0.827 | 0.847 |
| Highest precision among thresholds reaching ≥95% recall | 0.1170 | 0.481 | 0.962 | 0.641 |

These are validation results, not held-out test results. They apply to Logistic Regression's model probability, not the blended app risk score.

## Results

Run `train.py` to regenerate `metrics.json`. The held-out test results are written to the console and stored in the JSON report.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | 5-fold CV F1 (mean ± std) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Naive Bayes | 0.969 | 0.731 | 0.566 | 0.638 | 0.950 | 0.702 | 0.567 ± 0.049 |
| Logistic Regression | 0.993 | 0.951 | 0.896 | 0.923 | 0.992 | 0.953 | 0.842 ± 0.031 |
| Linear SVM (calibrated) | 0.994 | 0.981 | 0.896 | 0.937 | 0.993 | 0.958 | 0.857 ± 0.036 |
| SGD Classifier | 0.993 | 0.957 | 0.902 | 0.929 | 0.992 | 0.955 | 0.849 ± 0.037 |
| Random Forest* | 0.952 | 0.000 | 0.000 | 0.000 | 0.692 | 0.189 | 0.000 ± 0.000 |

These are the results from the current local run (Python 3.14, test probability cutoff 0.50). `*` The random forest row is the explicitly reported sparse compatibility fallback described below, not a CART Random Forest result. Its zero F1 means it predicted no fake postings at the default cutoff. Results depend on the dataset version, library versions, split, and implementation.

### Windows Application Control compatibility note

On some managed Windows installations, Application Control blocks the native DLL used by scikit-learn's `RandomForestClassifier`. The training script reports this explicitly and substitutes a lightweight randomized sparse-stump ensemble so that the other four models and the app remain usable. That fallback is **not a standard CART Random Forest**, and its result must not be presented as a standard Random Forest result. The saved metrics include the actual implementation name. If policy permits, use an approved Python/scikit-learn install with the tree extension enabled and rerun training for the standard forest comparison.

## Screenshots

Add genuine screenshots from a local run here, for example:

- `screenshots/dashboard-dark.png` — dashboard in dark theme
- `screenshots/analyze-result.png` — analysis evidence and rule checks
- `screenshots/model-lab.png` — held-out curves and CV comparison
- `screenshots/dashboard-light.png` — light theme

Do not include real applicants' personal information in screenshots.

## Run locally on Windows

Python 3.14 is supported by the project commands below. From PowerShell in `C:\jobguard`:

```powershell
py -3.14 -m pip install streamlit pandas numpy scikit-learn joblib plotly
py -3.14 train.py
py -3.14 -m streamlit run app.py
```

Training reads `fake_job_postings.csv` from the project directory and writes `model.joblib` and `metrics.json` there. The app expects both files beside `app.py`. User feedback is appended to the local `feedback.csv`; session history is kept in Streamlit session state.

## Screens and features

- Dashboard with training/test KPIs and session charts
- Single-posting analysis, model comparison, threshold sensitivity, red-flag rules, extraction, text statistics, and similar reference examples
- CSV batch scoring, session history, downloadable CSVs, and HTML report
- Model Lab with CV, ROC/PR, confusion matrices, and validation threshold analysis
- Explainability with LR terms, a what-if scorer, and qualitative review of test errors
- About and safety information, including India's cybercrime portal and helpline

## Ethics and responsible use

False positives can harm applicants and legitimate employers; false negatives can leave people exposed to scams. Do not use JobGuard as an automated hiring, candidate-screening, enforcement, or fraud-adjudication system. Verify organizations independently. Model confidence, TF-IDF similarity, regex flags, and explanations have important limitations and must not replace human judgment.

## Future work

See the project proposal and demo notes for multilingual evaluation, stronger temporal validation, a reviewed user-feedback process, model calibration analysis, and deployment hardening.
