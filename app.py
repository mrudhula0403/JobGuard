from __future__ import annotations

import html
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model.joblib"
METRICS_PATH = ROOT / "metrics.json"
FEEDBACK_PATH = ROOT / "feedback.csv"
MODEL_NAMES = (
    "Naive Bayes",
    "Logistic Regression",
    "Linear SVM (calibrated)",
    "SGD Classifier",
    "Random Forest",
)
SEVERITY_WEIGHTS = {"High": 3, "Medium": 2, "Low": 1}
PALETTE = {
    "blue": "#7689ff",
    "cyan": "#45c2cc",
    "green": "#55c69a",
    "yellow": "#f1bd64",
    "red": "#ef7383",
    "muted": "#97a2b5",
}
EXAMPLES = [
    {
        "label": "Remote data-entry offer",
        "category": "Likely fake · advance fee",
        "title": "Remote Data Entry Associate",
        "company": "Global Hiring Solutions",
        "salary": "₹8,000 per day",
        "email": "careers.hr@gmail.com",
        "link": "",
        "text": (
            "We are filling remote data-entry seats this week. No prior experience is "
            "needed, and the schedule is completely flexible. A refundable account "
            "activation payment of ₹1,500 is required before training begins. Send your "
            "receipt to our recruiter on WhatsApp to reserve the position."
        ),
    },
    {
        "label": "Overseas placement message",
        "category": "Likely fake · processing fee",
        "title": "International Hotel Placement",
        "company": "FastTrack Overseas",
        "salary": "₹2.5 lakh per month",
        "email": "placements@outlook.com",
        "link": "https://bit.ly/fasttrack-placement",
        "text": (
            "Selected candidates can begin an overseas hotel placement immediately. "
            "We arrange the visa interview, accommodation, and guaranteed monthly "
            "earnings. Pay the refundable document processing charge today so we can "
            "issue your appointment letter. Limited vacancies; message our placement "
            "coordinator directly for the payment details."
        ),
    },
    {
        "label": "Paid product internship",
        "category": "Likely legitimate · structured internship",
        "title": "Product Research Intern",
        "company": "Northstar Analytics",
        "salary": "₹24,000 per month",
        "email": "earlycareers@northstaranalytics.in",
        "link": "https://northstaranalytics.in/careers/product-research",
        "text": (
            "Northstar Analytics is hiring a Product Research Intern for a six-month "
            "paid placement in Bengaluru. You will assist with user interviews, "
            "organize survey results, and write research summaries with guidance from "
            "a product researcher. Applicants should be enrolled in an undergraduate "
            "program and have experience with spreadsheets. The internship is hybrid, "
            "offers a monthly stipend, and includes weekly mentoring. Candidates submit "
            "a resume through our careers page. Shortlisted students complete two "
            "interviews with the product team."
        ),
    },
    {
        "label": "Customer support role",
        "category": "Likely legitimate · established employer",
        "title": "Customer Support Associate",
        "company": "Cedar Retail India",
        "salary": "₹4.2–5 lakh per year",
        "email": "talent@cedarretail.in",
        "link": "https://cedarretail.in/jobs/support-associate",
        "text": (
            "Cedar Retail India is seeking a Customer Support Associate in Pune. The "
            "full-time role includes paid leave, product training, and a published "
            "annual salary range. You will resolve order questions, document cases, "
            "and coordinate with warehouse teams. Applicants should have strong "
            "written communication skills and be comfortable with rotating shifts. "
            "Apply through our careers portal; shortlisted candidates will speak with "
            "the support manager and recruiter."
        ),
    },
    {
        "label": "Software engineering role",
        "category": "Likely legitimate · technical hiring",
        "title": "Software Engineer",
        "company": "Riverbank Technologies",
        "salary": "₹12–18 lakh per year",
        "email": "engineering-hiring@riverbanktech.com",
        "link": "https://riverbanktech.com/company/careers",
        "text": (
            "Riverbank Technologies is looking for a Software Engineer to join our "
            "Bengaluru platform group. The role involves maintaining Python services, "
            "reviewing pull requests, and improving API reliability. We ask for two "
            "years of software experience, familiarity with SQL, and an interest in "
            "testing. Compensation and benefits are shared in the listing. Apply on "
            "our official careers site; shortlisted applicants complete a technical "
            "interview and a conversation with the team."
        ),
    },
    {
        "label": "Remote internship deposit",
        "category": "Likely fake · training deposit",
        "title": "Work-from-Home Marketing Intern",
        "company": "",
        "salary": "Earn ₹5,000 daily",
        "email": "internshipdesk@yahoo.com",
        "link": "",
        "text": (
            "Congratulations, your profile is approved for our marketing internship. "
            "There is no interview stage. Work from home and earn ₹5,000 every day by "
            "sharing product links. To unlock the starter kit, transfer a refundable "
            "training deposit before joining. Reply on Telegram today; seats close "
            "tonight."
        ),
    },
]


st.set_page_config(
    page_title="JobGuard | Trust before you apply",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner=False)
def load_artifacts(model_file: str, metrics_file: str) -> tuple[dict, dict]:
    """Load cached estimators and their training report from the project directory."""
    try:
        from train import SparseRandomForestFallback
    except ImportError:
        SparseRandomForestFallback = None
    if SparseRandomForestFallback is not None:
        setattr(sys.modules["__main__"], "SparseRandomForestFallback", SparseRandomForestFallback)
    bundle = joblib.load(model_file)
    with Path(metrics_file).open("r", encoding="utf-8") as stream:
        metrics = json.load(stream)
    if not isinstance(bundle, dict):
        raise ValueError("model.joblib must contain a dictionary.")
    models = bundle.get("models")
    if not isinstance(models, dict) or not models:
        if "vec" in bundle and "model" in bundle:
            raise ValueError(
                "This is an older single-model artifact. Run `py -3.14 train.py` "
                "to create the five-model bundle used by this version of JobGuard."
            )
        raise ValueError("The model artifact has no trained model pipelines.")
    missing = [name for name in MODEL_NAMES if name not in models]
    if missing:
        raise ValueError(f"model.joblib is missing model(s): {', '.join(missing)}.")
    missing_metrics = [name for name in MODEL_NAMES if name not in metrics]
    if missing_metrics:
        raise ValueError(
            f"metrics.json is missing model metrics for: {', '.join(missing_metrics)}. "
            "Run `py -3.14 train.py` again."
        )
    report_sections = (
        "cross_validation",
        "thresholds",
        "threshold_curves",
        "curves",
        "top_words",
        "misclassified_examples",
        "text_statistics_by_class",
    )
    missing_sections = [section for section in report_sections if section not in metrics]
    if missing_sections:
        raise ValueError(
            f"metrics.json is missing required analysis sections: {', '.join(missing_sections)}. "
            "Run `py -3.14 train.py` again."
        )
    return bundle, metrics


def initialize_state() -> None:
    """Create per-session UI state on first load."""
    defaults = {
        "theme": "dark",
        "navigation": "Dashboard",
        "history": [],
        "feedback_saved": set(),
        "selected_model": "Logistic Regression",
        "scan_results": {},
        "sample_to_load": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def apply_styles(theme: str) -> None:
    """Apply theme variables and responsive dashboard styling."""
    dark = theme == "dark"
    values = (
        {
            "bg": "#10131b",
            "sidebar": "#141823",
            "panel": "#191e2a",
            "panel_alt": "#202637",
            "text": "#edf1f8",
            "muted": "#a0aabd",
            "line": "#2b3242",
            "accent": "#8795ff",
            "shadow": "0 8px 26px rgba(0,0,0,.22)",
            "input": "#171c28",
            "good_bg": "#152c28",
            "warn_bg": "#342a1b",
            "bad_bg": "#352128",
        }
        if dark
        else {
            "bg": "#f4f6fb",
            "sidebar": "#ffffff",
            "panel": "#ffffff",
            "panel_alt": "#f7f8fc",
            "text": "#1c2432",
            "muted": "#667085",
            "line": "#e2e6ef",
            "accent": "#4f46e5",
            "shadow": "0 7px 24px rgba(29,42,74,.07)",
            "input": "#ffffff",
            "good_bg": "#eaf6f1",
            "warn_bg": "#fff4df",
            "bad_bg": "#ffedef",
        }
    )
    css_vars = ";".join(f"--jg-{key}:{value}" for key, value in values.items())
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        :root {{{css_vars};color-scheme:{'dark' if dark else 'light'};}}
        html,body,[class*="css"],[data-testid="stAppViewContainer"] {{
          font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;
        }}
        [data-testid="stHeader"],[data-testid="stToolbar"],#MainMenu,footer {{visibility:hidden;height:0;}}
        [data-testid="stAppViewContainer"] {{background:var(--jg-bg);color:var(--jg-text);}}
        [data-testid="stSidebar"] {{background:var(--jg-sidebar);border-right:1px solid var(--jg-line);}}
        [data-testid="stSidebar"] > div:first-child {{padding-top:1rem;}}
        .block-container {{max-width:1320px;padding:1.25rem 2rem 3rem;}}
        .brand {{display:flex;align-items:center;gap:10px;font-size:19px;font-weight:800;
          letter-spacing:-.6px;color:var(--jg-text);margin:0 0 3px;}}
        .brand-icon {{display:flex;align-items:center;justify-content:center;width:34px;height:34px;
          border-radius:11px;background:color-mix(in srgb,var(--jg-accent) 18%,transparent);
          color:var(--jg-accent);font-size:12px;font-weight:800;}}
        .brand-note {{margin:0 0 20px 44px;font-size:11px;color:var(--jg-muted);}}
        .topline {{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:3px;}}
        .page-title {{font-size:26px;font-weight:800;letter-spacing:-.8px;color:var(--jg-text);}}
        .page-subtitle {{font-size:13px;line-height:1.55;color:var(--jg-muted);margin:0 0 19px;}}
        .hero {{background:linear-gradient(118deg,#222849 0%,#373d79 52%,#485bb0 100%);
          border:1px solid rgba(255,255,255,.09);border-radius:20px;padding:25px 28px;
          box-shadow:0 12px 34px rgba(38,47,104,.23);margin:4px 0 20px;color:white;}}
        .hero-label {{font-size:10px;text-transform:uppercase;letter-spacing:1.35px;
          color:#c8ceff;font-weight:700;margin-bottom:8px;}}
        .hero-heading {{font-size:clamp(25px,4vw,36px);line-height:1.15;letter-spacing:-1.2px;
          font-weight:800;margin:0;}}
        .hero-copy {{font-size:13px;color:#e0e4ff;max-width:700px;line-height:1.6;margin:9px 0 16px;}}
        .hero-pills {{display:flex;flex-wrap:wrap;gap:8px;}}
        .hero-pill {{font-size:11px;padding:6px 10px;border:1px solid #ffffff38;
          border-radius:9px;background:#ffffff12;color:#f6f7ff;}}
        .card {{background:var(--jg-panel);border:1px solid var(--jg-line);border-radius:15px;
          padding:17px 18px;box-shadow:var(--jg-shadow);margin-bottom:13px;}}
        .card-heading {{font-size:14px;font-weight:700;color:var(--jg-text);margin:0 0 8px;}}
        .card-note {{font-size:11px;line-height:1.5;color:var(--jg-muted);}}
        .metric {{background:var(--jg-panel);border:1px solid var(--jg-line);border-radius:15px;
          padding:15px 16px;box-shadow:var(--jg-shadow);height:100%;}}
        .metric-value {{font-size:24px;line-height:1.2;font-weight:800;letter-spacing:-.6px;color:var(--jg-text);}}
        .metric-label {{font-size:11px;color:var(--jg-muted);margin-top:5px;}}
        .empty {{background:var(--jg-panel);border:1px dashed var(--jg-line);border-radius:15px;
          padding:28px 18px;text-align:center;color:var(--jg-muted);font-size:12px;line-height:1.6;}}
        .empty strong {{display:block;color:var(--jg-text);font-size:14px;margin-bottom:5px;}}
        .badge {{font-size:10px;font-weight:700;display:inline-block;padding:5px 9px;border-radius:999px;}}
        .badge-high {{background:var(--jg-bad-bg);color:#ef7383;}}
        .badge-medium {{background:var(--jg-warn-bg);color:#d99c3e;}}
        .badge-low {{background:var(--jg-good-bg);color:#42ae85;}}
        .rule-row {{display:flex;align-items:center;justify-content:space-between;gap:10px;
          border-bottom:1px solid var(--jg-line);padding:9px 0;}}
        .rule-row:last-child {{border-bottom:0;}}
        .rule-title {{font-size:12px;font-weight:600;color:var(--jg-text);}}
        .rule-detail {{font-size:10px;color:var(--jg-muted);margin-top:2px;}}
        .highlight-text {{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px;line-height:1.85;color:var(--jg-text);}}
        mark {{background:#f4d16b;color:#392b0a;padding:1px 3px;border-radius:3px;}}
        .note-box {{background:var(--jg-panel-alt);border:1px solid var(--jg-line);border-radius:11px;
          padding:12px 14px;color:var(--jg-muted);font-size:11px;line-height:1.6;}}
        .flow-row {{display:flex;align-items:center;gap:10px;flex-wrap:wrap;}}
        .flow-step {{padding:12px 14px;border:1px solid var(--jg-line);background:var(--jg-panel-alt);
          border-radius:11px;font-size:12px;font-weight:650;color:var(--jg-text);}}
        div.stButton > button,[data-testid="stDownloadButton"] button {{border-radius:10px;
          min-height:38px;font-weight:650;}}
        div.stButton > button[kind="primary"] {{background:var(--jg-accent);border-color:var(--jg-accent);}}
        [data-testid="stTextInput"] input,[data-testid="stTextArea"] textarea,
        [data-testid="stSelectbox"] [data-baseweb="select"] > div,
        [data-testid="stNumberInput"] input {{background:var(--jg-input);color:var(--jg-text);}}
        [data-testid="stDataFrame"],[data-testid="stTable"] {{border-radius:10px;overflow:hidden;}}
        [data-testid="stMarkdownContainer"] p,[data-testid="stMarkdownContainer"] li {{color:var(--jg-text);}}
        .stCaption,[data-testid="stCaptionContainer"] {{color:var(--jg-muted);}}
        @media(max-width:760px) {{
          .block-container {{padding:.9rem 1rem 2.2rem;}}
          .hero {{padding:21px 18px;border-radius:16px;}}
          .page-title {{font-size:22px;}}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def apply_plot_theme(figure: go.Figure) -> go.Figure:
    """Use matching Plotly palettes with readable hover and grid colors."""
    dark = st.session_state.theme == "dark"
    figure.update_layout(
        template="plotly_dark" if dark else "plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, sans-serif", "color": "#e8edf5" if dark else "#273043", "size": 11},
        margin={"l": 8, "r": 8, "t": 22, "b": 8},
        hoverlabel={"font": {"family": "Inter, sans-serif"}},
    )
    figure.update_xaxes(gridcolor="#30384b" if dark else "#e8ebf2", zerolinecolor="#424a5d" if dark else "#cbd2df")
    figure.update_yaxes(gridcolor="#30384b" if dark else "#e8ebf2", zerolinecolor="#424a5d" if dark else "#cbd2df")
    return figure


def model_threshold(metrics: dict, model_name: str, mode: str = "max_f1") -> float:
    try:
        return float(metrics["thresholds"][model_name][mode]["threshold"])
    except (KeyError, TypeError, ValueError):
        return float(metrics.get("default_threshold", 0.5))


def make_posting(fields: dict[str, str]) -> str:
    labels = (
        ("Job title", "title"),
        ("Company", "company"),
        ("Salary", "salary"),
        ("Contact email", "email"),
        ("Application link", "link"),
    )
    sections = [f"{label}: {fields[key].strip()}" for label, key in labels if fields.get(key, "").strip()]
    sections.append(fields.get("text", "").strip())
    return " ".join(value for value in sections if value)


def scan_rules(text: str, company: str, apply_link: str) -> list[dict]:
    """Return human-readable signals with severities and no opaque rule score."""
    email_text = " ".join(re.findall(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", text))
    all_text = f"{text}\n{company}\n{apply_link}\n{email_text}"
    urls = re.findall(r"(?:https?://|www\.)[^\s<>()]+", all_text, flags=re.I)
    free_domains = {
        "gmail.com", "yahoo.com", "yahoo.in", "outlook.com", "hotmail.com",
        "live.com", "rediffmail.com", "icloud.com",
    }
    free_email = any(
        address.rsplit("@", 1)[-1].lower() in free_domains
        for address in re.findall(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", all_text)
    )
    words = re.findall(r"\b[A-Za-z]{3,}\b", text)
    capitalized = re.findall(r"\b[A-Z]{4,}\b", text)
    uppercase_ratio = len(capitalized) / max(len(words), 1)
    lower_company = company.strip().lower()
    vague_company = not lower_company or any(
        phrase in lower_company for phrase in ("confidential", "undisclosed", "stealth", "private employer")
    )
    implausible_pay = bool(
        re.search(
            r"(?:₹|rs\.?\s*|inr\s*|\$\s*|usd\s*)\s*(?:[3-9]\d{3,}|\d+(?:\.\d+)?\s*(?:lakh|lac|crore))"
            r"\s*(?:per\s+day|daily|a\s+day)\b|"
            r"\b(?:earn|salary|income)\b.{0,40}\b\d[\d,]*\s*(?:per\s+day|daily)\b",
            all_text,
            re.I,
        )
    )
    rules = [
        ("Requests a fee, deposit, or payment", "High", bool(re.search(
            r"\b(registration|application|processing|training|security)\s+(fee|fees|charge|deposit)\b|"
            r"\b(pay|send|transfer)\b.{0,40}\b(fee|deposit|money|payment)\b",
            all_text, re.I,
        ))),
        ("Messaging-app-only contact", "High", bool(re.search(
            r"\b(only\s+(on|via|through)\s+(whats?app|telegram)|(whats?app|telegram)\s+only|"
            r"contact.{0,25}(only\s+)?(on|via|through)\s+(whats?app|telegram))\b",
            all_text, re.I,
        ))),
        ("Free email used as company contact", "Medium", free_email),
        ("No interview mentioned", "Medium", bool(re.search(r"\b(no|without)\s+(job\s+)?interview\b", all_text, re.I))),
        ("Urgency or scarcity language", "Low", bool(re.search(
            r"\b(urgent|hurry|limited vacancies|limited spots|apply now|act now|last chance|today only|"
            r"immediate joining|join immediately|joining today)\b",
            all_text, re.I,
        ))),
        ("Unrealistic daily-pay claim", "High", implausible_pay),
        ("Excessive ALL-CAPS wording", "Low", len(capitalized) >= 4 and uppercase_ratio >= 0.25),
        ("Company identity is vague or absent", "Medium", vague_company),
        ("Shortened or suspicious application URL", "High", any(
            re.search(r"(bit\.ly|tinyurl\.com|t\.co|goo\.gl|is\.gd|cutt\.ly|shorturl\.at|"
                      r"rebrand\.ly|ow\.ly|rb\.gy|ipfs\.io)", url, re.I)
            or bool(re.search(r"https?://(?:\d{1,3}\.){3}\d{1,3}", url, re.I))
            for url in urls
        )),
    ]
    return [
        {"name": name, "severity": severity, "flagged": flagged}
        for name, severity, flagged in rules
    ]


def rule_score(flags: list[dict]) -> float:
    total = sum(SEVERITY_WEIGHTS[item["severity"]] for item in flags)
    found = sum(
        SEVERITY_WEIGHTS[item["severity"]]
        for item in flags
        if item["flagged"]
    )
    return found / max(total, 1)


def get_probability(estimator: Any, text: str) -> float:
    """Get fake-class probability, including models that expose only decision scores."""
    if hasattr(estimator, "predict_proba"):
        probabilities = estimator.predict_proba([text])[0]
        class_ids = list(estimator.classes_)
        return float(probabilities[class_ids.index(1)])
    if hasattr(estimator, "decision_function"):
        score = float(estimator.decision_function([text])[0])
        return float(1 / (1 + np.exp(-np.clip(score, -40, 40))))
    raise ValueError("Selected model cannot return a probability or decision score.")


def score_posting(
    bundle: dict, metrics: dict, text: str, company: str, apply_link: str
) -> dict:
    models = bundle["models"]
    probabilities = {name: get_probability(models[name], text) for name in MODEL_NAMES}
    selected = st.session_state.selected_model
    flags = scan_rules(text, company, apply_link)
    rules = rule_score(flags)
    probability = probabilities[selected]
    combined = 0.75 * probability + 0.25 * rules
    threshold = float(st.session_state.get("risk_threshold", model_threshold(metrics, selected)))
    if combined >= threshold:
        verdict, level = "Likely Fake", "high"
    elif combined >= threshold * 0.55:
        verdict, level = "Suspicious", "medium"
    else:
        verdict, level = "Looks Legitimate", "low"
    explanations = logistic_contributions(bundle, text)
    return {
        "id": uuid4().hex,
        "time": datetime.now().astimezone(),
        "text": text,
        "company": company,
        "apply_link": apply_link,
        "selected_model": selected,
        "probabilities": probabilities,
        "probability": probability,
        "rule_score": rules,
        "risk_score": combined,
        "threshold": threshold,
        "verdict": verdict,
        "level": level,
        "threshold_margin": "Wide" if abs(combined - threshold) >= 0.25 else (
            "Moderate" if abs(combined - threshold) >= 0.10 else "Narrow"
        ),
        "flags": flags,
        "terms": explanations,
        "entities": extract_entities(text, company),
        "text_stats": text_statistics(text),
        "similar": similar_postings(bundle, text),
    }


def logistic_contributions(bundle: dict, text: str) -> list[tuple[str, float]]:
    """Explain terms with the LR coefficients regardless of selected scoring model."""
    lr = bundle["models"]["Logistic Regression"]
    vectorizer = lr.named_steps["tfidf"]
    classifier = lr.named_steps["classifier"]
    matrix = vectorizer.transform([text]).tocsr()
    contributions = matrix.data * classifier.coef_[0, matrix.indices]
    terms = vectorizer.get_feature_names_out()
    positive = [
        (str(terms[index]), float(value))
        for index, value in zip(matrix.indices, contributions)
        if value > 0
    ]
    return sorted(positive, key=lambda item: item[1], reverse=True)[:12]


def text_statistics(text: str) -> dict[str, float]:
    words = re.findall(r"\b[\w'-]+\b", text)
    sentences = max(1, len(re.findall(r"[.!?]+", text)))
    vowel_groups = sum(
        max(1, len(re.findall(r"[aeiouy]+", word.lower().strip("'-"))))
        for word in words
        if word.strip("'-")
    )
    if words:
        vowel_groups -= sum(word.lower().endswith("e") and len(word) > 2 for word in words)
    letters = [char for char in text if char.isalpha()]
    syllables = max(len(words), vowel_groups)
    return {
        "word_count": float(len(words)),
        "readability": float(206.835 - 1.015 * len(words) / sentences - 84.6 * syllables / max(len(words), 1)),
        "uppercase_ratio": float(sum(char.isupper() for char in letters) / max(len(letters), 1)),
        "exclamation_count": float(text.count("!")),
    }


def extract_entities(text: str, company: str) -> list[dict]:
    """Collect contact and compensation entities using transparent regex patterns."""
    found: list[dict] = []
    for address in sorted(set(re.findall(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", text))):
        domain = address.rsplit("@", 1)[-1].lower()
        suspicious = domain in {
            "gmail.com", "yahoo.com", "yahoo.in", "outlook.com", "hotmail.com",
            "live.com", "rediffmail.com", "icloud.com",
        }
        found.append({"type": "Email", "value": address, "suspicious": suspicious})
    for phone in sorted(set(re.findall(r"(?<!\w)(?:\+?\d[\d\s().-]{8,}\d)(?!\w)", text))):
        cleaned = re.sub(r"\D", "", phone)
        if 10 <= len(cleaned) <= 15:
            found.append({"type": "Phone", "value": phone.strip(), "suspicious": False})
    urls = sorted(set(re.findall(r"(?:https?://|www\.)[^\s<>()]+", text, re.I)))
    for url in urls:
        suspicious = bool(re.search(
            r"(bit\.ly|tinyurl\.com|t\.co|goo\.gl|is\.gd|cutt\.ly|shorturl\.at|rebrand\.ly|ow\.ly|rb\.gy)",
            url, re.I,
        ))
        found.append({"type": "URL", "value": url.rstrip(".,;"), "suspicious": suspicious})
    amounts = sorted(set(re.findall(
        r"(?:₹|rs\.?\s*|inr\s*|\$\s*|usd\s*)\s*\d[\d,]*(?:\.\d+)?\s*(?:lakh|lac|crore)?"
        r"(?:\s*(?:per\s+year|per\s+month|per\s+day|annually|monthly|daily))?",
        text, re.I,
    )))
    suspicious_pay = any(
        flag["flagged"] and flag["name"] == "Unrealistic daily-pay claim"
        for flag in scan_rules(text, company, "")
    )
    for amount in amounts:
        found.append({"type": "Money", "value": amount.strip(), "suspicious": suspicious_pay})
    names = {company.strip()} if company.strip() else set()
    names.update(re.findall(
        r"\b(?:at|from|with)\s+([A-Z][A-Za-z0-9&.'-]*(?:\s+[A-Z][A-Za-z0-9&.'-]*){0,3})",
        text,
    ))
    vague_names = {"confidential", "undisclosed", "stealth", "private employer"}
    for name in sorted(names):
        found.append({
            "type": "Company (NER-lite)",
            "value": name,
            "suspicious": name.strip().lower() in vague_names,
        })
    return found


def similar_postings(bundle: dict, text: str) -> list[dict]:
    reference = bundle.get("fake_reference")
    if not reference or reference.get("count", 0) == 0:
        return []
    vectorizer = bundle["models"]["Logistic Regression"].named_steps["tfidf"]
    query = vectorizer.transform([text])
    scores = cosine_similarity(query, reference["vectors"]).ravel()
    best = np.argsort(scores)[-3:][::-1]
    return [
        {"similarity": float(scores[index]), "text": reference["snippets"][index]}
        for index in best
    ]


def render_page_header(title: str, subtitle: str) -> None:
    st.markdown(f"<div class='page-title'>{html.escape(title)}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='page-subtitle'>{html.escape(subtitle)}</div>", unsafe_allow_html=True)


def metric_card(label: str, value: str, note: str = "") -> None:
    st.markdown(
        f"<div class='metric'><div class='metric-value'>{html.escape(value)}</div>"
        f"<div class='metric-label'>{html.escape(label)}"
        f"{(' · ' + html.escape(note)) if note else ''}</div></div>",
        unsafe_allow_html=True,
    )


def empty_state(title: str, message: str) -> None:
    st.markdown(
        f"<div class='empty'><strong>{html.escape(title)}</strong>{html.escape(message)}</div>",
        unsafe_allow_html=True,
    )


def chart_layout(fig: go.Figure, height: int = 300) -> go.Figure:
    apply_plot_theme(fig)
    fig.update_layout(height=height)
    return fig


def render_dashboard(metrics: dict) -> None:
    render_page_header("Dashboard", "A live overview of model quality and your screening session.")
    best_model = max(MODEL_NAMES, key=lambda name: metrics[name]["f1"])
    best = metrics[best_model]
    dataset = metrics.get("dataset", {})
    rows = int(dataset.get("rows", 17_880))
    kpi_cols = st.columns(4)
    kpis = [
        ("Postings trained on", f"{rows:,}", "EMSCAD"),
        ("Best test F1", f"{float(best['f1']):.1%}", best_model),
        ("Best fake recall", f"{float(best['recall']):.1%}", best_model),
        ("Best ROC-AUC", f"{float(best['roc_auc']):.3f}", best_model),
    ]
    for column, (label, value, note) in zip(kpi_cols, kpis):
        with column:
            metric_card(label, value, note)
    st.write("")
    history = st.session_state.history
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("<div class='card-heading'>Screening mix · this session</div>", unsafe_allow_html=True)
        if not history:
            empty_state("No scans yet", "Analyze a posting to start your session dashboard.")
        else:
            counts = pd.Series([entry["verdict"] for entry in history]).value_counts()
            fig = px.pie(
                names=counts.index,
                values=counts.values,
                hole=0.68,
                color_discrete_sequence=[PALETTE["red"], PALETTE["yellow"], PALETTE["green"]],
            )
            fig.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(chart_layout(fig, 300), width="stretch", config={"displayModeBar": False})
    with right:
        st.markdown("<div class='card-heading'>Risk score over time</div>", unsafe_allow_html=True)
        if len(history) < 2:
            empty_state("Trend appears after two scans", "Your session risk scores are shown here in time order.")
        else:
            frame = pd.DataFrame(history).sort_values("time")
            fig = px.line(
                frame,
                x="time",
                y="risk_score",
                markers=True,
                color_discrete_sequence=[PALETTE["blue"]],
                hover_data={"verdict": True, "snippet": True, "risk_score": ":.1%"},
            )
            fig.update_yaxes(tickformat=".0%", range=[0, 1], title="Combined risk")
            fig.update_xaxes(title="")
            st.plotly_chart(chart_layout(fig, 300), width="stretch", config={"displayModeBar": False})
    st.markdown("<div class='card-heading'>Quick start</div>", unsafe_allow_html=True)
    action_cols = st.columns(3)
    for index, (column, label) in enumerate(zip(action_cols, ("Analyze a posting", "Scan a CSV", "Explore the models"))):
        with column:
            if st.button(label, key=f"quick_{index}", use_container_width=True):
                st.session_state.pending_page = ("Analyze", "Batch Scan", "Model Lab")[index]
                if index == 0:
                    st.session_state.sample_to_load = 0
                st.rerun()
    metric_card("Scans this session", str(len(history)), "Stored in this browser session")


def plot_gauge(probability: float, risk: float, model_name: str) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "font": {"size": 32}},
            title={"text": f"{model_name} · fake probability"},
            gauge={
                "axis": {"range": [0, 100], "ticksuffix": "%"},
                "bar": {"color": PALETTE["blue"], "thickness": 0.23},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 35], "color": "#15372f" if st.session_state.theme == "dark" else "#e8f5ef"},
                    {"range": [35, 60], "color": "#3a3020" if st.session_state.theme == "dark" else "#fff3dc"},
                    {"range": [60, 100], "color": "#3a242a" if st.session_state.theme == "dark" else "#ffeaec"},
                ],
                "threshold": {
                    "line": {"color": PALETTE["yellow"], "width": 3},
                    "thickness": 0.8,
                    "value": risk * 100,
                },
            },
        )
    )
    fig.update_layout(
        height=250,
        margin={"l": 15, "r": 15, "t": 36, "b": 3},
        transition={"duration": 650, "easing": "cubic-in-out"},
    )
    return chart_layout(fig, 250)


def render_rules(flags: list[dict]) -> None:
    rows = []
    for flag in flags:
        status = "Flag found" if flag["flagged"] else "Clear"
        level = flag["severity"].lower()
        rows.append(
            f"<div class='rule-row'><div><div class='rule-title'>{html.escape(flag['name'])}</div>"
            f"<div class='rule-detail'>Severity: {html.escape(flag['severity'])}</div></div>"
            f"<span class='badge badge-{level}'>{status}</span></div>"
        )
    st.markdown(
        "<div class='card'><div class='card-heading'>Red-flag rule checks</div>"
        + "".join(rows)
        + "</div>",
        unsafe_allow_html=True,
    )


def render_highlighted(text: str, terms: list[tuple[str, float]]) -> None:
    escaped = html.escape(text)
    unigram_terms = sorted({word for word, _ in terms if " " not in word}, key=len, reverse=True)
    if unigram_terms:
        pattern = re.compile(r"\b(" + "|".join(map(re.escape, unigram_terms)) + r")\b", re.I)
        escaped = pattern.sub(lambda match: f"<mark>{match.group(0)}</mark>", escaped)
    st.markdown(
        "<div class='card'><div class='card-heading'>Posting terms highlighted</div>"
        f"<div class='highlight-text'>{escaped}</div></div>",
        unsafe_allow_html=True,
    )


def report_html(result: dict) -> str:
    flags = "".join(
        f"<li>{html.escape(item['name'])} — "
        f"{'flag found' if item['flagged'] else 'clear'} ({item['severity']})</li>"
        for item in result["flags"]
    )
    terms = ", ".join(html.escape(term) for term, _ in result["terms"]) or "No strong positive LR terms"
    posting = html.escape(result["text"])
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width"><title>JobGuard report</title>
<style>body{{font:15px/1.6 Inter,Arial,sans-serif;max-width:850px;margin:40px auto;padding:0 22px;color:#20283a}}
h1{{color:#4338ca}}section{{border:1px solid #e2e6ef;border-radius:14px;padding:18px;margin:16px 0}}
.score{{font-size:28px;font-weight:800}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}.muted{{color:#667085}}</style></head>
<body><h1>JobGuard screening report</h1><p class="muted">Generated {html.escape(result['time'].isoformat(timespec='minutes'))}</p>
<section><div class="score">{html.escape(result['verdict'])}</div><p>Combined risk: <b>{result['risk_score']:.1%}</b></p>
<p>Model: {html.escape(result['selected_model'])} · Model probability: {result['probability']:.1%} · Rule score: {result['rule_score']:.1%}</p>
<p>Decision threshold: {result['threshold']:.1%}</p></section>
<section><h2>Rule checks</h2><ul>{flags}</ul></section>
<section><h2>Top fake-associated terms (Logistic Regression)</h2><p>{terms}</p></section>
<section><h2>Analyzed posting</h2><pre>{posting}</pre></section>
<p class="muted">Screening aid, not proof of fraud. Verify employers and offers independently.</p></body></html>"""


def append_history(result: dict) -> None:
    st.session_state.history.insert(
        0,
        {
            "time": result["time"],
            "snippet": result["text"].replace("\n", " ")[:180],
            "verdict": result["verdict"],
            "probability": result["probability"],
            "risk_score": result["risk_score"],
            "model": result["selected_model"],
            "id": result["id"],
        },
    )
    st.session_state.history = st.session_state.history[:500]


def record_feedback(result: dict, response: str) -> None:
    existing = set(st.session_state.feedback_saved)
    key = (result["id"], response)
    if key in existing:
        st.info("Feedback for this response is already recorded.")
        return
    item = pd.DataFrame([{
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "posting_snippet": result["text"].replace("\n", " ")[:400],
        "model": result["selected_model"],
        "model_probability": result["probability"],
        "combined_risk": result["risk_score"],
        "verdict": result["verdict"],
        "feedback": response,
    }])
    try:
        item.to_csv(
            FEEDBACK_PATH,
            mode="a",
            header=not FEEDBACK_PATH.exists(),
            index=False,
            encoding="utf-8",
        )
    except OSError as error:
        st.error(f"Could not save feedback.csv: {error}")
        return
    existing.add(key)
    st.session_state.feedback_saved = existing
    st.success("Feedback saved locally. Thank you.")


def render_result(result: dict, metrics: dict) -> None:
    status_descriptions = {
        "Likely Fake": "Several signals are consistent with risky postings. Verify independently before responding.",
        "Suspicious": "Some signals need a closer look. Confirm the recruiter and role through official channels.",
        "Looks Legitimate": "No strong combined signals were found. This is not a guarantee of legitimacy.",
    }
    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        st.markdown(
            f"<div class='card'><span class='badge badge-{result['level']}'>{html.escape(result['verdict'])}</span>"
                f"<div style='font-size:19px;font-weight:800;margin-top:9px'>{html.escape(result['threshold_margin'])} threshold margin</div>"
            f"<div class='card-note'>{html.escape(status_descriptions[result['verdict']])}</div>"
            f"<div style='margin-top:13px;font-size:12px;color:var(--jg-muted)'>"
            f"Blended risk <b style='color:var(--jg-text)'>{result['risk_score']:.1%}</b> · "
            f"Model probability <b style='color:var(--jg-text)'>{result['probability']:.1%}</b> · "
            f"Rule score <b style='color:var(--jg-text)'>{result['rule_score']:.1%}</b></div></div>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            plot_gauge(result["probability"], result["risk_score"], result["selected_model"]),
            width="stretch",
            config={"displayModeBar": False, "responsive": True},
        )
    with c2:
        st.markdown("<div class='card-heading'>All model probabilities</div>", unsafe_allow_html=True)
        probability_frame = pd.DataFrame(
            [{"Model": name, "Fake probability": score} for name, score in result["probabilities"].items()]
        )
        st.dataframe(
            probability_frame.sort_values("Fake probability", ascending=False),
            width="stretch",
            hide_index=True,
            column_config={"Fake probability": st.column_config.ProgressColumn(
                "Fake probability", min_value=0.0, max_value=1.0, format="%.1%%"
            )},
        )
        st.markdown(
            "<div class='note-box'><b>Blended risk formula:</b> "
            "75% selected model fake probability + 25% severity-weighted rule score. "
            "The gauge shows the selected model’s probability; the marker shows blended risk. "
            "The verdict threshold is adjustable in the sidebar.</div>",
            unsafe_allow_html=True,
        )

    if result["terms"]:
        words = pd.DataFrame(result["terms"], columns=["Term", "Contribution"]).sort_values("Contribution")
        fig = px.bar(words, x="Contribution", y="Term", orientation="h", color_discrete_sequence=[PALETTE["red"]])
        fig.update_layout(xaxis_title="Positive LR contribution toward fake class", yaxis_title="")
        st.markdown("<div class='card-heading'>Why this result? · Logistic Regression feature contributions</div>", unsafe_allow_html=True)
        st.plotly_chart(chart_layout(fig, max(250, 33 * len(words))), width="stretch", config={"displayModeBar": False})
        render_highlighted(result["text"], result["terms"])
    else:
        st.info("No active vocabulary terms with positive Logistic Regression fake-class contribution were found.")

    rule_col, info_col = st.columns(2, gap="large")
    with rule_col:
        render_rules(result["flags"])
    with info_col:
        st.markdown("<div class='card-heading'>Information extraction · NER-lite</div>", unsafe_allow_html=True)
        if result["entities"]:
            frame = pd.DataFrame(result["entities"])
            st.dataframe(frame, width="stretch", hide_index=True)
        else:
            empty_state("No entities found", "No email addresses, phone numbers, URLs, money amounts, or company names were detected.")
        st.markdown("<div class='card-heading' style='margin-top:12px'>Text profile</div>", unsafe_allow_html=True)
        render_text_profile(result["text_stats"], metrics.get("text_statistics_by_class", {}))

    st.markdown("<div class='card-heading'>Most similar known scam postings</div>", unsafe_allow_html=True)
    similar = result["similar"]
    if not similar:
        empty_state("Reference set unavailable", "Train JobGuard to save example fake postings for similarity search.")
    else:
        for index, item in enumerate(similar, 1):
            with st.expander(f"Reference {index} · cosine similarity {item['similarity']:.1%}"):
                st.write(item["text"])
        st.caption("Similarity is based on TF-IDF cosine similarity to labeled fake postings in the training split; it is not a probability.")

    feedback_cols = st.columns([0.8, 1.2, 2.5])
    with feedback_cols[0]:
        if st.button("Correct", key=f"feedback_yes_{result['id']}", use_container_width=True):
            record_feedback(result, "Correct")
    with feedback_cols[1]:
        if st.button("Incorrect", key=f"feedback_no_{result['id']}", use_container_width=True):
            record_feedback(result, "Incorrect")
    with feedback_cols[2]:
        st.download_button(
            "Download HTML report",
            data=report_html(result),
            file_name="jobguard-report.html",
            mime="text/html",
            key=f"report_{result['id']}",
            use_container_width=True,
        )


def render_text_profile(stats: dict[str, float], class_stats: dict) -> None:
    labels = {
        "word_count": ("Words", ".0f"),
        "readability": ("Readability estimate", ".1f"),
        "uppercase_ratio": ("Uppercase ratio", ".1%"),
        "exclamation_count": ("Exclamation marks", ".0f"),
    }
    rows = []
    for key, (label, fmt) in labels.items():
        value = float(stats.get(key, 0.0))
        real = float(class_stats.get("real", {}).get(key, 0.0))
        fake = float(class_stats.get("fake", {}).get(key, 0.0))
        rows.append({
            "Statistic": label,
            "This posting": format(value, fmt),
            "Training real avg.": format(real, fmt),
            "Training fake avg.": format(fake, fmt),
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.caption("Readability is a lightweight Flesch-style estimate; class averages are descriptive, not decision rules.")


def render_analyze(bundle: dict, metrics: dict) -> None:
    render_page_header(
        "Analyze a posting",
        "Screen the text, compare models, and inspect the evidence behind the risk score.",
    )
    if st.session_state.sample_to_load is not None:
        sample = EXAMPLES[int(st.session_state.sample_to_load)]
        for field in ("title", "company", "salary", "email", "link", "text"):
            st.session_state[f"posting_{field}"] = sample.get(field, "")
        st.session_state.sample_to_load = None

    with st.expander("Load a demo example", expanded=False):
        st.caption("Six different sample postings across advance-fee, internship, overseas, customer-support, and software roles.")
        example_cols = st.columns(3)
        for index, sample in enumerate(EXAMPLES):
            with example_cols[index % len(example_cols)]:
                if st.button(sample["label"], key=f"example_{index}", use_container_width=True):
                    st.session_state.sample_to_load = index
                    st.rerun()
                st.caption(sample["category"])

    form_col, result_col = st.columns([1.02, 0.98], gap="large")
    with form_col:
        st.markdown("<div class='card-heading'>Posting details</div>", unsafe_allow_html=True)
        st.text_input("Job title · optional", key="posting_title")
        st.text_input("Company · optional", key="posting_company")
        st.text_area(
            "Posting text",
            key="posting_text",
            height=230,
            placeholder="Paste the complete job or internship posting here…",
        )
        with st.expander("Additional details · optional"):
            st.text_input("Salary", key="posting_salary")
            st.text_input("Contact email", key="posting_email")
            st.text_input("Apply link", key="posting_link")
        if st.button("Analyze posting", type="primary", key="run_analysis", use_container_width=True):
            if len(st.session_state.posting_text.strip()) < 25:
                st.session_state.pop("last_result", None)
                st.error("Please enter at least 25 characters of posting text.")
            else:
                text = make_posting({
                    "title": st.session_state.posting_title,
                    "company": st.session_state.posting_company,
                    "salary": st.session_state.posting_salary,
                    "email": st.session_state.posting_email,
                    "link": st.session_state.posting_link,
                    "text": st.session_state.posting_text,
                })
                with st.spinner("Comparing models and checking posting signals…"):
                    try:
                        result = score_posting(
                            bundle,
                            metrics,
                            text,
                            st.session_state.posting_company,
                            st.session_state.posting_link,
                        )
                    except (ValueError, KeyError, AttributeError) as error:
                        st.error(f"Could not score this posting: {error}")
                    else:
                        result["text_stats"] = text_statistics(st.session_state.posting_text)
                        st.session_state.last_result = result
                        append_history(result)

    with result_col:
        result = st.session_state.get("last_result")
        if result:
            render_result(result, metrics)
        else:
            empty_state(
                "Your analysis will appear here",
                "Enter a posting or load an example, then run Analyze posting.",
            )


def render_batch(bundle: dict) -> None:
    render_page_header("Batch Scan", "Analyze a spreadsheet of postings using your selected model and sensitivity.")
    file = st.file_uploader("Upload CSV · requires a text column", type=["csv"], key="batch_upload")
    if file is None:
        empty_state("Ready for a CSV", "Optional columns: title, company, salary, email, link. Each row needs at least 25 characters in text.")
        return
    try:
        source = pd.read_csv(file)
    except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as error:
        st.error(f"Could not read this CSV: {error}")
        return
    if "text" not in source.columns:
        st.error("This CSV must contain a column named `text`.")
        return
    if source.empty:
        st.warning("The CSV contains no rows.")
        return
    invalid = source["text"].fillna("").astype(str).str.strip().str.len() < 25
    if invalid.any():
        st.error(f"{int(invalid.sum())} row(s) have empty or very short text. Fix them and upload again.")
        return
    if len(source) > 5000:
        st.error("For responsive batch scanning, upload at most 5,000 postings per file.")
        return
    with st.spinner(f"Scoring {len(source):,} postings…"):
        rows = []
        selected = st.session_state.selected_model
        threshold = float(st.session_state.risk_threshold)
        for _, row in source.iterrows():
            company_value = row.get("company", "")
            link_value = row.get("link", "")
            company = "" if pd.isna(company_value) else str(company_value)
            link = "" if pd.isna(link_value) else str(link_value)
            text = make_posting({
                "title": "" if pd.isna(row.get("title", "")) else str(row.get("title", "")),
                "company": company,
                "salary": "" if pd.isna(row.get("salary", "")) else str(row.get("salary", "")),
                "email": "" if pd.isna(row.get("email", "")) else str(row.get("email", "")),
                "link": link,
                "text": str(row["text"]),
            })
            try:
                probability = get_probability(bundle["models"][selected], text)
            except (ValueError, AttributeError) as error:
                st.error(f"Could not score the batch: {error}")
                return
            flags = scan_rules(text, company, link)
            rules = rule_score(flags)
            risk = 0.75 * probability + 0.25 * rules
            verdict = "Likely Fake" if risk >= threshold else (
                "Suspicious" if risk >= threshold * 0.55 else "Looks Legitimate"
            )
            result_row = row.to_dict()
            result_row.update({
                "model": selected,
                "model_probability": round(probability, 4),
                "rule_score": round(rules, 4),
                "risk_score": round(risk, 4),
                "verdict": verdict,
                "red_flags": "; ".join(flag["name"] for flag in flags if flag["flagged"]),
            })
            rows.append(result_row)
    result_frame = pd.DataFrame(rows)
    st.success(f"Scanned {len(result_frame):,} postings.")
    summary_cols = st.columns(3)
    for column, verdict in zip(summary_cols, ("Likely Fake", "Suspicious", "Looks Legitimate")):
        with column:
            metric_card(verdict, str(int((result_frame["verdict"] == verdict).sum())))
    fig = px.histogram(
        result_frame,
        x="risk_score",
        color="verdict",
        nbins=20,
        color_discrete_map={
            "Likely Fake": PALETTE["red"],
            "Suspicious": PALETTE["yellow"],
            "Looks Legitimate": PALETTE["green"],
        },
    )
    fig.update_xaxes(tickformat=".0%", title="Blended risk score")
    st.plotly_chart(chart_layout(fig, 300), width="stretch", config={"displayModeBar": False})
    display = result_frame.sort_values("risk_score", ascending=False)
    st.dataframe(
        display,
        width="stretch",
        hide_index=True,
        column_config={
            "model_probability": st.column_config.NumberColumn("Model probability", format="%.1%%"),
            "rule_score": st.column_config.NumberColumn("Rule score", format="%.1%%"),
            "risk_score": st.column_config.ProgressColumn("Risk score", min_value=0, max_value=1, format="%.1%%"),
        },
    )
    st.download_button(
        "Download results CSV",
        result_frame.to_csv(index=False).encode("utf-8-sig"),
        "jobguard-batch-results.csv",
        "text/csv",
        use_container_width=True,
    )


def render_history() -> None:
    render_page_header("History", "Session-only record of postings you have scanned.")
    history = st.session_state.history
    if not history:
        empty_state("No scans yet", "Analyze postings to create a session history.")
        return
    c1, c2, c3 = st.columns([1.1, 1.1, 2])
    verdicts = ["All"] + sorted({row["verdict"] for row in history})
    with c1:
        selected_verdict = st.selectbox("Filter by verdict", verdicts)
    with c2:
        minimum_risk = st.slider(
            "Minimum risk · 0 to 1",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
            format="%.2f",
            help="Filter out postings below this blended risk score.",
        )
    with c3:
        search = st.text_input("Search posting snippets")
    filtered = [
        row for row in history
        if (selected_verdict == "All" or row["verdict"] == selected_verdict)
        and row["risk_score"] >= minimum_risk
        and search.casefold() in row["snippet"].casefold()
    ]
    action, _ = st.columns([1, 4])
    with action:
        if st.button("Clear session history", key="clear_history"):
            st.session_state.history = []
            st.rerun()
    if not filtered:
        empty_state("No matching scans", "Adjust the filters to see other results.")
        return
    frame = pd.DataFrame(filtered).drop(columns=["id"], errors="ignore")
    st.dataframe(
        frame,
        width="stretch",
        hide_index=True,
        column_config={
            "probability": st.column_config.NumberColumn("Model probability", format="%.1%%"),
            "risk_score": st.column_config.ProgressColumn("Combined risk", min_value=0, max_value=1, format="%.1%%"),
        },
    )
    st.download_button(
        "Download filtered history",
        frame.to_csv(index=False).encode("utf-8-sig"),
        "jobguard-history.csv",
        "text/csv",
    )


def render_model_lab(metrics: dict) -> None:
    render_page_header("Model Lab", "Compare held-out performance, cross-validation stability, and operating thresholds.")
    scores = pd.DataFrame([
        {
            "Model": name,
            "Implementation": metrics[name].get("implementation", name),
            "Accuracy": metrics[name]["accuracy"],
            "Precision": metrics[name]["precision"],
            "Recall": metrics[name]["recall"],
            "F1": metrics[name]["f1"],
            "ROC-AUC": metrics[name]["roc_auc"],
            "PR-AUC": metrics[name]["pr_auc"],
            "CV F1 mean": metrics["cross_validation"][name]["f1_mean"],
            "CV F1 std": metrics["cross_validation"][name]["f1_std"],
        }
        for name in MODEL_NAMES
    ])
    st.dataframe(scores, width="stretch", hide_index=True, column_config={
        column: st.column_config.NumberColumn(column, format="%.3f")
        for column in scores.columns if column != "Model"
        and column != "Implementation"
    })
    implementation = metrics["Random Forest"].get("implementation", "RandomForestClassifier")
    if implementation != "RandomForestClassifier":
        st.warning(
            f"Random Forest compatibility note: this run used `{implementation}` because Windows Application Control "
            "blocked scikit-learn's tree DLL. This is not a standard CART random forest; do not describe it as one."
        )
    compare = scores.melt(
        id_vars="Model",
        value_vars=["Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"],
        var_name="Metric",
        value_name="Score",
    )
    fig = px.bar(
        compare,
        x="Metric",
        y="Score",
        color="Model",
        barmode="group",
        color_discrete_sequence=[PALETTE["blue"], PALETTE["cyan"], PALETTE["yellow"], PALETTE["green"], PALETTE["red"]],
        range_y=[0, 1],
    )
    st.plotly_chart(chart_layout(fig, 380), width="stretch", config={"displayModeBar": False})

    cv = metrics["cross_validation"]
    cv_fig = go.Figure()
    for name in MODEL_NAMES:
        cv_fig.add_trace(go.Bar(
            name=name,
            x=[name],
            y=[cv[name]["f1_mean"]],
            error_y={"type": "data", "array": [cv[name]["f1_std"]], "visible": True},
        ))
    cv_fig.update_layout(showlegend=False, yaxis_title="5-fold F1 (mean ± std)", yaxis_range=[0, 1])
    st.markdown("<div class='card-heading'>Cross-validation stability · train partition only</div>", unsafe_allow_html=True)
    st.plotly_chart(chart_layout(cv_fig, 310), width="stretch", config={"displayModeBar": False})

    roc_fig = go.Figure()
    pr_fig = go.Figure()
    for name in MODEL_NAMES:
        curve = metrics["curves"][name]
        roc_fig.add_trace(go.Scatter(
            x=curve["roc_fpr"], y=curve["roc_tpr"], mode="lines", name=f"{name} · AUC {metrics[name]['roc_auc']:.3f}"
        ))
        pr_fig.add_trace(go.Scatter(
            x=curve["pr_recall"], y=curve["pr_precision"], mode="lines", name=f"{name} · AP {metrics[name]['pr_auc']:.3f}"
        ))
    roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line={"dash": "dash"}, name="Random baseline"))
    roc_fig.update_layout(xaxis_title="False positive rate", yaxis_title="True positive rate")
    pr_fig.update_layout(xaxis_title="Recall", yaxis_title="Precision")
    roc_col, pr_col = st.columns(2, gap="large")
    with roc_col:
        st.markdown("<div class='card-heading'>ROC curves · held-out test</div>", unsafe_allow_html=True)
        st.plotly_chart(chart_layout(roc_fig, 370), width="stretch", config={"displayModeBar": False})
    with pr_col:
        st.markdown("<div class='card-heading'>Precision–recall curves · held-out test</div>", unsafe_allow_html=True)
        st.plotly_chart(chart_layout(pr_fig, 370), width="stretch", config={"displayModeBar": False})

    st.markdown("<div class='card-heading'>Confusion matrices · held-out test at 0.50</div>", unsafe_allow_html=True)
    confusion_model = st.selectbox("Select confusion matrix", MODEL_NAMES, key="confusion_model")
    matrix = metrics[confusion_model]["confusion"]
    heatmap = go.Figure(go.Heatmap(
        z=matrix,
        x=["Predicted real", "Predicted fake"],
        y=["Actual real", "Actual fake"],
        text=matrix,
        texttemplate="%{text}",
        textfont={"size": 20},
        colorscale="Blues",
        showscale=False,
    ))
    st.plotly_chart(chart_layout(heatmap, 260), width="stretch", config={"displayModeBar": False})

    tradeoff_model = st.selectbox("Threshold trade-off model", MODEL_NAMES, key="tradeoff_model")
    operating_threshold = st.slider(
        "Validation threshold",
        0.0,
        1.0,
        0.5,
        0.01,
        key="lab_threshold",
        help="Validation-set operating point. Thresholds were selected without using held-out test labels.",
    )
    candidates = metrics.get("threshold_curves", {}).get(tradeoff_model, [])
    selected_point = min(candidates, key=lambda point: abs(point["threshold"] - operating_threshold)) if candidates else None
    if selected_point:
        op_cols = st.columns(3)
        for column, label, key in zip(op_cols, ("Precision", "Recall", "F1"), ("precision", "recall", "f1")):
            with column:
                metric_card(label, f"{selected_point[key]:.1%}", f"nearest stored validation point · threshold {selected_point['threshold']:.2f}")
    thresholds = metrics.get("thresholds", {}).get(tradeoff_model, {})
    st.markdown(
        "<div class='note-box'><b>Threshold recommendations from validation:</b> "
        f"Max-F1 threshold {float(thresholds.get('max_f1', {}).get('threshold', 0.5)):.3f}; "
        f"recall ≥95% threshold {float(thresholds.get('recall_at_least_95', {}).get('threshold', 0.0)):.3f}. "
        "If validation could not attain 95% recall, the report marks the target as unmet.</div>",
        unsafe_allow_html=True,
    )

    always_real = float(metrics["always_real_accuracy"])
    st.warning(
        f"Accuracy alone is misleading for this imbalanced dataset: a model that predicts every listing as real achieves "
        f"{always_real:.1%} accuracy while detecting zero fake postings. Inspect fake-class recall, precision, F1, and PR-AUC."
    )
    st.info(
        "Linear SVM probabilities use sigmoid calibration. Probability calibration is not a guarantee of real-world calibration; "
        "calibration drift should be monitored before deployment."
    )


def render_explainability(bundle: dict, metrics: dict) -> None:
    render_page_header("Explainability", "Inspect global model signals, edit a posting, and review test-set errors.")
    words = metrics.get("top_words", {})
    fake = words.get("fake", [])
    real = words.get("real", [])
    fake_col, real_col = st.columns(2, gap="large")
    for column, title, entries, color in (
        (fake_col, "Top terms associated with fake", fake, PALETTE["red"]),
        (real_col, "Top terms associated with real", real, PALETTE["green"]),
    ):
        with column:
            st.markdown(f"<div class='card-heading'>{title} · LR coefficients</div>", unsafe_allow_html=True)
            if entries:
                frame = pd.DataFrame(entries)
                frame["coefficient"] = frame["coefficient"].abs()
                frame = frame.sort_values("coefficient")
                fig = px.bar(frame, x="coefficient", y="term", orientation="h", color_discrete_sequence=[color])
                fig.update_layout(xaxis_title="Absolute coefficient magnitude", yaxis_title="")
                st.plotly_chart(chart_layout(fig, 560), width="stretch", config={"displayModeBar": False})
            else:
                empty_state("No terms available", "Run training to save global Logistic Regression coefficients.")

    st.markdown("<div class='card-heading'>What-if analysis</div>", unsafe_allow_html=True)
    st.caption("Changes are rescored immediately with the selected sidebar model. Local feature explanations remain LR-based.")
    what_if = st.text_area(
        "Edit a posting to explore how its score changes",
        key="what_if_text",
        height=190,
        placeholder="Paste or edit a posting here…",
    )
    if len(what_if.strip()) >= 25:
        try:
            simulation = score_posting(bundle, metrics, what_if, "", "")
        except (ValueError, KeyError, AttributeError) as error:
            st.error(f"Could not score this text: {error}")
        else:
            cols = st.columns(4)
            for column, label, value in zip(
                cols,
                ("Selected model probability", "Rule score", "Combined risk", "Verdict"),
                (
                    f"{simulation['probability']:.1%}",
                    f"{simulation['rule_score']:.1%}",
                    f"{simulation['risk_score']:.1%}",
                    simulation["verdict"],
                ),
            ):
                with column:
                    metric_card(label, value)
    else:
        empty_state("Enter a little more text", "The what-if scorer activates at 25 characters.")

    st.markdown("<div class='card-heading'>Misclassified held-out examples · Logistic Regression</div>", unsafe_allow_html=True)
    errors = metrics.get("misclassified_examples", [])
    if not errors:
        empty_state("No error examples saved", "The test split may have had no errors or the training report is from an older format.")
    else:
        frame = pd.DataFrame(errors)
        frame["true_label"] = frame["true_label"].map({0: "Real", 1: "Fake"})
        frame["predicted_label"] = frame["predicted_label"].map({0: "Predicted real", 1: "Predicted fake"})
        frame = frame.rename(columns={
            "text_snippet": "Text snippet",
            "true_label": "Actual",
            "predicted_label": "Model prediction",
            "fake_probability": "Fake probability",
            "confidence": "Prediction confidence",
            "commentary": "Error-analysis note",
        })
        st.dataframe(frame, width="stretch", hide_index=True, column_config={
            "Fake probability": st.column_config.NumberColumn(format="%.1%%"),
            "Prediction confidence": st.column_config.NumberColumn(format="%.1%%"),
        })
        st.caption(
            "False negatives may use reassuring language or omit obvious scam signals; false positives may contain "
            "terms the model associates with fake ads. Snippets support qualitative error analysis, not causal claims."
        )


def render_about(metrics: dict) -> None:
    render_page_header("About & Safety", "How JobGuard works, what the evaluation means, and how to respond to a suspected scam.")
    st.markdown(
        "<div class='flow-row'><div class='flow-step'>Posting input</div><span>→</span>"
        "<div class='flow-step'>Text + fields</div><span>→</span><div class='flow-step'>TF-IDF</div>"
        "<span>→</span><div class='flow-step'>Five classifiers</div><span>→</span>"
        "<div class='flow-step'>Rules + explanations</div></div>",
        unsafe_allow_html=True,
    )
    dataset = metrics.get("dataset", {})
    st.markdown("<div class='card-heading' style='margin-top:20px'>Dataset and evaluation</div>", unsafe_allow_html=True)
    st.markdown(
        f"Trained on **{int(dataset.get('rows', 17_880)):,} EMSCAD job postings** "
        f"({int(dataset.get('fake_count', 0)):,} fake, {int(dataset.get('real_count', 0)):,} real). "
        "The provided text fields are joined and represented with TF-IDF unigrams and bigrams. "
        "A stratified 80/20 split is used; model selection includes five-fold CV on the training partition, "
        "while decision thresholds are selected on a separate validation subset."
    )
    st.markdown(
        "<div class='note-box'><b>Ethics and limitations:</b> EMSCAD is an older, English-language dataset and may not "
        "represent current hiring scams, local labor markets, or multilingual postings. Text cues can encode bias, "
        "and legitimate employers may use unusual wording. Similarity and rules are heuristic aids. Do not use this score "
        "to deny someone a job or make legal, financial, or safety decisions without independent verification.</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='card-heading' style='margin-top:18px'>Safety checklist</div>", unsafe_allow_html=True)
    for item in (
        "Never pay an application, interview, training, placement, or equipment fee to secure a job.",
        "Find the employer’s official website yourself and verify the opening on its careers page.",
        "Check the recruiter’s email domain and independently published company phone number.",
        "Do not share OTPs, bank passwords, identity documents, or sensitive information with an unverified contact.",
        "Be cautious of pressure to act immediately or communicate exclusively through messaging apps.",
    ):
        st.markdown(f"- {item}")
    st.markdown("<div class='card-heading' style='margin-top:18px'>If you think you have been scammed</div>", unsafe_allow_html=True)
    st.markdown(
        "Stop contact, preserve emails/messages/URLs and payment receipts, and contact your bank or payment provider immediately. "
        "In India, report the incident at [cybercrime.gov.in](https://cybercrime.gov.in) or call the national cybercrime helpline **1930** "
        "as soon as possible. Contact local emergency services if you are in immediate danger."
    )
    st.warning("JobGuard is a screening aid, not proof of fraud. Verify every company, recruiter, and offer independently.")


def render_sidebar() -> tuple[str, str]:
    pending_page = st.session_state.pop("pending_page", None)
    if pending_page in ("Dashboard", "Analyze", "Batch Scan", "History", "Model Lab", "Explainability", "About & Safety"):
        st.session_state.navigation = pending_page
    st.sidebar.markdown(
        "<div class='brand'><span class='brand-icon'>JG</span>JobGuard</div>"
        "<div class='brand-note'>Trust before you apply</div>",
        unsafe_allow_html=True,
    )
    page_options = (
        "Dashboard", "Analyze", "Batch Scan", "History", "Model Lab", "Explainability", "About & Safety"
    )
    page = st.sidebar.radio(
        "Workspace",
        page_options,
        key="navigation",
        format_func=lambda item: {
            "Dashboard": "⌂  Dashboard",
            "Analyze": "◎  Analyze",
            "Batch Scan": "▤  Batch Scan",
            "History": "◷  History",
            "Model Lab": "▦  Model Lab",
            "Explainability": "⌕  Explainability",
            "About & Safety": "◇  About & Safety",
        }[item],
    )
    model_name = st.sidebar.selectbox(
        "Scoring model",
        MODEL_NAMES,
        key="selected_model",
        help="All model probabilities are shown when you analyze a posting.",
    )
    if "risk_threshold" not in st.session_state:
        st.session_state.risk_threshold = 0.50
    threshold = st.sidebar.slider(
        "Sensitivity · decision threshold",
        min_value=0.05,
        max_value=0.95,
        step=0.01,
        key="risk_threshold",
        help="A lower threshold is more sensitive and labels more postings as risky.",
        format="%.2f",
    )
    st.sidebar.caption(f"Likely Fake at {threshold:.0%} blended risk or above.")
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Scans this session: **{len(st.session_state.history)}**")
    st.sidebar.caption("Research prototype · screening aid, not proof")
    return page, model_name


def refresh_existing_result(bundle: dict, metrics: dict, model_name: str, threshold: float) -> None:
    """Keep an existing result and its history row consistent with current controls."""
    previous = st.session_state.get("last_result")
    if not previous or (
        previous["selected_model"] == model_name
        and abs(previous["threshold"] - threshold) < 1e-9
    ):
        return
    refreshed = score_posting(
        bundle,
        metrics,
        previous["text"],
        previous["company"],
        previous["apply_link"],
    )
    refreshed["id"] = previous["id"]
    refreshed["time"] = previous["time"]
    st.session_state.last_result = refreshed
    for history_row in st.session_state.history:
        if history_row["id"] == previous["id"]:
            history_row.update({
                "verdict": refreshed["verdict"],
                "probability": refreshed["probability"],
                "risk_score": refreshed["risk_score"],
                "model": refreshed["selected_model"],
            })
            break


def main() -> None:
    initialize_state()
    if not MODEL_PATH.is_file():
        st.error("JobGuard is missing `model.joblib`. Run `py -3.14 train.py` first.")
        st.stop()
    if not METRICS_PATH.is_file():
        st.error("JobGuard is missing `metrics.json`. Run `py -3.14 train.py` first.")
        st.stop()
    try:
        bundle, metrics = load_artifacts(str(MODEL_PATH), str(METRICS_PATH))
    except (OSError, ValueError, KeyError, EOFError, json.JSONDecodeError) as error:
        st.error(f"Could not load JobGuard training artifacts: {error}")
        st.stop()
    page, model_name = render_sidebar()
    refresh_existing_result(
        bundle,
        metrics,
        model_name,
        float(st.session_state.risk_threshold),
    )

    top_col, toggle_col = st.columns([5, 1], vertical_alignment="center")
    with top_col:
        st.markdown(
            "<div class='topline'><div class='card-note'>JOBGUARD · POSTING SAFETY INTELLIGENCE</div></div>",
            unsafe_allow_html=True,
        )
    with toggle_col:
        new_theme = st.toggle("Light theme", value=st.session_state.theme == "light", key="theme_toggle")
        chosen_theme = "light" if new_theme else "dark"
        if chosen_theme != st.session_state.theme:
            st.session_state.theme = chosen_theme
            st.rerun()
    apply_styles(st.session_state.theme)

    if page == "Dashboard":
        st.markdown(
            "<div class='hero'><div class='hero-label'>AI-assisted job-posting review</div>"
            "<div class='hero-heading'>Spot suspicious offers before you apply.</div>"
            "<div class='hero-copy'>Compare five text classifiers, review practical red flags, and see what drove a screening result.</div>"
            f"<div class='hero-pills'><span class='hero-pill'>EMSCAD research baseline</span>"
            f"<span class='hero-pill'>5 models · stratified evaluation</span>"
            f"<span class='hero-pill'>Selected model · {html.escape(model_name)}</span></div></div>",
            unsafe_allow_html=True,
        )
        render_dashboard(metrics)
    elif page == "Analyze":
        render_analyze(bundle, metrics)
    elif page == "Batch Scan":
        render_batch(bundle)
    elif page == "History":
        render_history()
    elif page == "Model Lab":
        render_model_lab(metrics)
    elif page == "Explainability":
        render_explainability(bundle, metrics)
    else:
        render_about(metrics)
    st.caption("JobGuard is an educational screening aid—not proof of fraud, legal advice, or a substitute for verification.")


if __name__ == "__main__":
    main()
