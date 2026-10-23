"""Train and evaluate JobGuard's five text classifiers.

Run from the project directory with:
    py -3.14 train.py

The held-out test split is kept separate from cross-validation and threshold
selection. TF-IDF vectorizers are refit inside every cross-validation fold.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "fake_job_postings.csv"
MODEL_PATH = ROOT / "model.joblib"
METRICS_PATH = ROOT / "metrics.json"
SEED = 42
TEST_SIZE = 0.20
THRESHOLD_VALIDATION_SIZE = 0.15
FAKE_REFERENCE_COUNT = 100
CURVE_POINTS = 300

TEXT_COLUMNS = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits",
]
MODEL_NAMES = (
    "Naive Bayes",
    "Logistic Regression",
    "Linear SVM (calibrated)",
    "SGD Classifier",
    "Random Forest",
)


class SparseRandomForestFallback(ClassifierMixin, BaseEstimator):
    """Sparse randomized stump ensemble for environments blocking sklearn tree DLLs.

    This is an explicitly reported compatibility fallback, not the standard
    CART-based sklearn RandomForestClassifier.
    """

    def __init__(self, n_estimators: int = 100, random_state: int = SEED):
        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit(self, X: Any, y: Any) -> "SparseRandomForestFallback":
        labels = np.asarray(y, dtype=np.int8)
        self.classes_ = np.array([0, 1], dtype=np.int8)
        rng = np.random.default_rng(self.random_state)
        self.n_features_in_ = int(X.shape[1])
        self.prior_ = (float(labels.sum()) + 1.0) / (len(labels) + 2.0)
        self.stumps_: list[tuple[int, float, float]] = []
        for _ in range(self.n_estimators):
            feature = int(rng.integers(0, self.n_features_in_))
            values = np.asarray(X[:, feature].toarray()).ravel()
            present = values > 0
            count_present = int(present.sum())
            count_absent = len(labels) - count_present
            if count_present == 0 or count_absent == 0:
                self.stumps_.append((feature, self.prior_, self.prior_))
                continue
            present_rate = (float(labels[present].sum()) + 1.0) / (count_present + 2.0)
            absent_rate = (float(labels[~present].sum()) + 1.0) / (count_absent + 2.0)
            self.stumps_.append((feature, present_rate, absent_rate))
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        predictions = np.empty((X.shape[0], len(self.stumps_)), dtype=np.float32)
        for column, (feature, present_rate, absent_rate) in enumerate(self.stumps_):
            values = np.asarray(X[:, feature].toarray()).ravel()
            predictions[:, column] = np.where(values > 0, present_rate, absent_rate)
        fake_probability = predictions.mean(axis=1)
        return np.column_stack((1.0 - fake_probability, fake_probability))

    def predict(self, X: Any) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(np.int8)


if __name__ == "__main__":
    sys.modules.setdefault("train", sys.modules[__name__])
    SparseRandomForestFallback.__module__ = "train"


def read_dataset() -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Load and validate EMSCAD data, producing the project text field."""
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}. Put fake_job_postings.csv beside train.py."
        )
    frame = pd.read_csv(DATA_PATH)
    missing = sorted(set(TEXT_COLUMNS + ["fraudulent"]) - set(frame.columns))
    if missing:
        raise ValueError(f"The dataset is missing required columns: {', '.join(missing)}")
    text = frame[TEXT_COLUMNS].fillna("").astype(str).agg(" ".join, axis=1).str.strip()
    labels = pd.to_numeric(frame["fraudulent"], errors="raise").astype(int)
    if set(labels.unique()) != {0, 1}:
        raise ValueError("The 'fraudulent' column must contain binary 0/1 labels.")
    if text.str.len().lt(1).any():
        raise ValueError("The dataset contains empty posting text.")
    return text, labels, frame


def vectorizer(max_features: int = 100_000) -> TfidfVectorizer:
    """Return the consistent unigram/bigram representation for each model."""
    return TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=3,
        max_features=max_features,
        sublinear_tf=True,
        dtype=np.float32,
    )


def build_models() -> dict[str, Pipeline]:
    """Create model pipelines; calibrated SVM exposes comparable probabilities."""
    try:
        from sklearn.ensemble import RandomForestClassifier

        forest: Any = RandomForestClassifier(
            n_estimators=100,
            max_depth=24,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=SEED,
        )
    except ImportError as error:
        print(
            "WARNING: sklearn RandomForestClassifier is unavailable "
            f"({error}). Using a sparse randomized-stump compatibility ensemble. "
            "Fix the Windows Application Control policy to use the CART random forest.",
            flush=True,
        )
        forest = SparseRandomForestFallback(n_estimators=100, random_state=SEED)

    estimators = {
        "Naive Bayes": ComplementNB(alpha=0.5),
        "Logistic Regression": LogisticRegression(
            max_iter=1600,
            class_weight="balanced",
            C=5,
            solver="liblinear",
            random_state=SEED,
        ),
        "Linear SVM (calibrated)": CalibratedClassifierCV(
            estimator=LinearSVC(class_weight="balanced", C=1.0, random_state=SEED),
            method="sigmoid",
            cv=3,
            ensemble=False,
        ),
        "SGD Classifier": SGDClassifier(
            loss="log_loss",
            alpha=1e-5,
            max_iter=1500,
            tol=1e-3,
            class_weight="balanced",
            random_state=SEED,
        ),
        "Random Forest": forest,
    }
    pipelines = {}
    for name, estimator in estimators.items():
        feature_limit = 30_000 if name == "Random Forest" else 100_000
        pipelines[name] = Pipeline(
            [
                ("tfidf", vectorizer(feature_limit)),
                ("classifier", estimator),
            ]
        )
    return pipelines


def compact_curve(values: np.ndarray, limit: int = CURVE_POINTS) -> list[float]:
    """Downsample a curve without changing its endpoints to keep JSON compact."""
    if len(values) <= limit:
        return [float(value) for value in values]
    indices = np.unique(np.linspace(0, len(values) - 1, limit).astype(int))
    return [float(values[index]) for index in indices]


def tune_thresholds(
    y_true: np.ndarray, probabilities: np.ndarray
) -> tuple[dict[str, Any], list[dict[str, float]]]:
    """Tune operating points on validation data, never on the held-out test set."""
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if not len(thresholds):
        return (
            {
                "max_f1": {"threshold": 0.5, "precision": 0.0, "recall": 0.0, "f1": 0.0},
                "recall_at_least_95": {
                    "threshold": 0.0,
                    "precision": 0.0,
                    "recall": 1.0,
                    "f1": 0.0,
                },
            },
            [],
        )

    p = precision[:-1]
    r = recall[:-1]
    f1_values = 2 * p * r / np.maximum(p + r, 1e-12)
    best_f1_index = int(np.argmax(f1_values))

    eligible = np.flatnonzero(r >= 0.95)
    if len(eligible):
        best_recall_index = int(eligible[np.argmax(p[eligible])])
        high_recall = {
            "threshold": float(thresholds[best_recall_index]),
            "precision": float(p[best_recall_index]),
            "recall": float(r[best_recall_index]),
            "f1": float(f1_values[best_recall_index]),
        }
    else:
        high_recall = {
            "threshold": 0.0,
            "precision": float(precision[0]),
            "recall": float(recall[0]),
            "f1": float(
                2 * precision[0] * recall[0] / max(precision[0] + recall[0], 1e-12)
            ),
            "target_met": False,
        }

    thresholds_report = {
        "max_f1": {
            "threshold": float(thresholds[best_f1_index]),
            "precision": float(p[best_f1_index]),
            "recall": float(r[best_f1_index]),
            "f1": float(f1_values[best_f1_index]),
        },
        "recall_at_least_95": high_recall,
    }
    validation_curve = [
        {
            "threshold": float(thresholds[index]),
            "precision": float(p[index]),
            "recall": float(r[index]),
            "f1": float(f1_values[index]),
        }
        for index in np.unique(np.linspace(0, len(thresholds) - 1, min(200, len(thresholds))).astype(int))
    ]
    return thresholds_report, validation_curve


def evaluate(
    y_true: np.ndarray, probabilities: np.ndarray
) -> tuple[dict[str, Any], dict[str, list[float]]]:
    """Calculate fixed-0.5 test metrics and compact ROC/PR curve coordinates."""
    predictions = (probabilities >= 0.5).astype(int)
    fpr, tpr, _ = roc_curve(y_true, probabilities)
    precision, recall, _ = precision_recall_curve(y_true, probabilities)
    pr_auc = auc(recall[::-1], precision[::-1])
    scores = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(pr_auc),
        "confusion": confusion_matrix(y_true, predictions, labels=[0, 1]).tolist(),
    }
    curves = {
        "roc_fpr": compact_curve(fpr),
        "roc_tpr": compact_curve(tpr),
        "pr_recall": compact_curve(recall),
        "pr_precision": compact_curve(precision),
    }
    return scores, curves


def count_syllables(word: str) -> int:
    """Estimate English syllables with a small dependency-free heuristic."""
    word = re.sub(r"[^a-z]", "", word.lower())
    if not word:
        return 0
    groups = re.findall(r"[aeiouy]+", word)
    count = len(groups)
    if word.endswith("e") and count > 1 and not word.endswith(("le", "ye")):
        count -= 1
    return max(1, count)


def text_statistics(texts: pd.Series) -> dict[str, float]:
    """Compute simple document-level summaries to contextualize the dataset."""
    word_counts: list[int] = []
    readability: list[float] = []
    uppercase_ratios: list[float] = []
    exclamations: list[int] = []
    for raw_text in texts.fillna("").astype(str):
        words = re.findall(r"\b[\w'-]+\b", raw_text)
        sentences = max(1, len(re.findall(r"[.!?]+", raw_text)))
        syllables = sum(count_syllables(word) for word in words)
        word_counts.append(len(words))
        readability.append(
            206.835
            - 1.015 * (len(words) / sentences)
            - 84.6 * (syllables / max(len(words), 1))
        )
        letters = [character for character in raw_text if character.isalpha()]
        uppercase_ratios.append(
            sum(character.isupper() for character in letters) / max(len(letters), 1)
        )
        exclamations.append(raw_text.count("!"))
    return {
        "word_count": float(np.mean(word_counts)) if word_counts else 0.0,
        "readability": float(np.mean(readability)) if readability else 0.0,
        "uppercase_ratio": float(np.mean(uppercase_ratios)) if uppercase_ratios else 0.0,
        "exclamation_count": float(np.mean(exclamations)) if exclamations else 0.0,
    }


def top_logistic_terms(
    fitted_model: Pipeline, count: int = 30
) -> dict[str, list[dict[str, Any]]]:
    """Extract highest positive and negative logistic-regression coefficients."""
    fitted_vectorizer = fitted_model.named_steps["tfidf"]
    estimator = fitted_model.named_steps["classifier"]
    coefficients = estimator.coef_[0]
    terms = fitted_vectorizer.get_feature_names_out()
    fake_ids = np.argsort(coefficients)[-count:][::-1]
    real_ids = np.argsort(coefficients)[:count]
    return {
        "fake": [
            {"term": str(terms[index]), "coefficient": float(coefficients[index])}
            for index in fake_ids
        ],
        "real": [
            {"term": str(terms[index]), "coefficient": float(coefficients[index])}
            for index in real_ids
        ],
    }


def misclassified_examples(
    texts: pd.Series,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    count: int = 15,
) -> list[dict[str, Any]]:
    """Return up to 15 test errors, prioritizing confident mistakes."""
    predictions = (probabilities >= 0.5).astype(int)
    wrong = np.flatnonzero(predictions != y_true)
    wrong = sorted(wrong, key=lambda index: abs(float(probabilities[index]) - 0.5), reverse=True)
    return [
        {
            "text_snippet": str(texts.iloc[index])[:600],
            "true_label": int(y_true[index]),
            "predicted_label": int(predictions[index]),
            "fake_probability": float(probabilities[index]),
            "confidence": float(max(probabilities[index], 1 - probabilities[index])),
            "commentary": (
                "False negative: fake-labeled posting scored below 0.50. Review for scam language "
                "that may be absent from the model's learned lexical cues."
                if y_true[index] == 1
                else "False positive: real-labeled posting scored at or above 0.50. Check for "
                "legitimate recruiting language that overlaps with learned fake-associated terms."
            ),
        }
        for index in wrong[:count]
    ]


def create_fake_reference(
    fitted_lr: Pipeline,
    train_text: pd.Series,
    train_labels: pd.Series,
) -> dict[str, Any]:
    """Create compact, reusable TF-IDF vectors for known fake training examples."""
    fake_positions = np.flatnonzero(train_labels.to_numpy() == 1)
    rng = np.random.default_rng(SEED)
    selected = rng.choice(
        fake_positions,
        size=min(FAKE_REFERENCE_COUNT, len(fake_positions)),
        replace=False,
    )
    selected_texts = train_text.iloc[selected].astype(str).tolist()
    fitted_vectorizer = fitted_lr.named_steps["tfidf"]
    sparse_vectors = fitted_vectorizer.transform(selected_texts)
    return {
        "vectors": sparse_vectors.tocsr(),
        "snippets": [text[:500] for text in selected_texts],
        "count": len(selected_texts),
    }


def main() -> None:
    started = time.perf_counter()
    texts, labels, _ = read_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=TEST_SIZE,
        stratify=labels,
        random_state=SEED,
    )
    X_fit, X_tune, y_fit, y_tune = train_test_split(
        X_train,
        y_train,
        test_size=THRESHOLD_VALIDATION_SIZE,
        stratify=y_train,
        random_state=SEED,
    )
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    pipelines = build_models()
    fitted_models: dict[str, Pipeline] = {}
    model_metrics: dict[str, dict[str, Any]] = {}
    model_cv: dict[str, dict[str, Any]] = {}
    model_thresholds: dict[str, dict[str, Any]] = {}
    model_curves: dict[str, dict[str, list[float]]] = {}
    model_threshold_curves: dict[str, list[dict[str, float]]] = {}
    test_probabilities: dict[str, np.ndarray] = {}

    print(f"Dataset: {len(texts):,} postings; fake rate: {labels.mean():.2%}")
    print(
        f"Split: {len(X_train):,} train / {len(X_test):,} held-out test; "
        f"{len(X_fit):,} model-fit / {len(X_tune):,} threshold validation"
    )
    print("Running five-fold stratified CV on the model-fit partition ...", flush=True)

    for name, model_pipeline in pipelines.items():
        model_started = time.perf_counter()
        cv_values = cross_val_score(
            model_pipeline,
            X_fit,
            y_fit,
            cv=folds,
            scoring="f1",
            n_jobs=1,
            error_score="raise",
        )
        model_cv[name] = {
            "f1_mean": float(np.mean(cv_values)),
            "f1_std": float(np.std(cv_values, ddof=1)),
            "fold_f1": [float(value) for value in cv_values],
        }

        threshold_model = build_models()[name]
        threshold_model.fit(X_fit, y_fit)
        tune_probabilities = threshold_model.predict_proba(X_tune)[:, 1]
        model_thresholds[name], model_threshold_curves[name] = tune_thresholds(
            y_tune.to_numpy(), tune_probabilities
        )

        model_pipeline.fit(X_train, y_train)
        probabilities = model_pipeline.predict_proba(X_test)[:, 1]
        test_probabilities[name] = probabilities
        model_metrics[name], model_curves[name] = evaluate(y_test.to_numpy(), probabilities)
        estimator = model_pipeline.named_steps["classifier"]
        model_metrics[name]["implementation"] = type(estimator).__name__
        fitted_models[name] = model_pipeline
        print(
            f"  {name:<25} CV F1 {model_cv[name]['f1_mean']:.3f} ± "
            f"{model_cv[name]['f1_std']:.3f} | test F1 "
            f"{model_metrics[name]['f1']:.3f} | "
            f"{time.perf_counter() - model_started:.1f}s",
            flush=True,
        )

    always_real_accuracy = float((y_test.to_numpy() == 0).mean())
    lr = fitted_models["Logistic Regression"]
    lr_probabilities = test_probabilities["Logistic Regression"]
    mistakes = misclassified_examples(
        X_test.reset_index(drop=True),
        y_test.to_numpy(),
        lr_probabilities,
    )
    class_stats = {
        "real": text_statistics(X_train[y_train.to_numpy() == 0]),
        "fake": text_statistics(X_train[y_train.to_numpy() == 1]),
    }
    coefficients = top_logistic_terms(lr)
    reference = create_fake_reference(lr, X_train.reset_index(drop=True), y_train.reset_index(drop=True))

    metrics: dict[str, Any] = {
        **model_metrics,
        "cross_validation": model_cv,
        "thresholds": model_thresholds,
        "threshold_curves": model_threshold_curves,
        "curves": model_curves,
        "top_words": coefficients,
        "misclassified_examples": mistakes,
        "text_statistics_by_class": class_stats,
        "always_real_accuracy": always_real_accuracy,
        "dataset": {
            "name": "EMSCAD / fake_job_postings.csv",
            "rows": int(len(texts)),
            "fake_count": int(labels.sum()),
            "real_count": int((labels == 0).sum()),
            "fake_rate": float(labels.mean()),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "threshold_validation_rows": int(len(X_tune)),
            "random_state": SEED,
            "test_size": TEST_SIZE,
        },
        "selected_model": "Logistic Regression",
        "default_threshold": 0.5,
        "threshold_selection_note": (
            "Operating thresholds were selected using a separate stratified validation "
            "split from training data. Test metrics and curves use a separate held-out "
            "test set at the default 0.5 threshold."
        ),
    }

    # Keep old model.joblib consumers working while exposing every trained estimator.
    bundle = {
        "vec": lr.named_steps["tfidf"],
        "model": lr.named_steps["classifier"],
        "models": fitted_models,
        "selected_model": "Logistic Regression",
        "fake_reference": reference,
        "feature_names": lr.named_steps["tfidf"].get_feature_names_out(),
    }
    temporary_model = MODEL_PATH.with_suffix(".joblib.tmp")
    temporary_metrics = METRICS_PATH.with_suffix(".json.tmp")
    joblib.dump(bundle, temporary_model, compress=3)
    with temporary_metrics.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)
    temporary_model.replace(MODEL_PATH)
    temporary_metrics.replace(METRICS_PATH)

    table = pd.DataFrame(
        [
            {
                "Model": name,
                "Implementation": model_metrics[name]["implementation"],
                "CV F1 (mean ± std)": (
                    f"{model_cv[name]['f1_mean']:.3f} ± {model_cv[name]['f1_std']:.3f}"
                ),
                "Accuracy": model_metrics[name]["accuracy"],
                "Precision": model_metrics[name]["precision"],
                "Recall": model_metrics[name]["recall"],
                "F1": model_metrics[name]["f1"],
                "ROC-AUC": model_metrics[name]["roc_auc"],
                "PR-AUC": model_metrics[name]["pr_auc"],
            }
            for name in MODEL_NAMES
        ]
    )
    print("\nHeld-out test results (probability threshold = 0.50):")
    print(table.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nAlways-predict-real test accuracy: {always_real_accuracy:.3%}")
    print(f"LR test errors saved: {len(mistakes)}")
    print(f"Fake reference postings saved: {reference['count']}")
    print(f"Saved {MODEL_PATH.name} and {METRICS_PATH.name}")
    print(f"Total training time: {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
