# ==========================================================
# Fake News Detection using AI  (+ live "select & scan" API)
# ==========================================================
# Models: calibrated TF-IDF Logistic Regression, Naive Bayes, Linear SVM,
# SGD Classifier, Ridge, and a Bidirectional LSTM. Dataset CSV columns:
# text,label  where 0 = Real and 1 = Fake.
#
# Usage:
#   pip install -r requirements.txt
#   python FakeNewsDetection.py train --csv FakeNews.csv --no-plots
#   python FakeNewsDetection.py serve
#   python FakeNewsDetection.py predict --text "some news" --url https://example.com
#
# IMPORTANT: retrain after replacing this file (new cleaning, padding, files).
# ==========================================================

import os
import re
import sys
import json
import html
import traceback
import argparse
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import joblib
import mlflow

from text_checks import check_text_usable, news_likeness, MAX_CHARS
try:
    from text_checks import check_prediction_length
except ImportError:
    def check_prediction_length(text, minimum=100):
        """Fallback used when text_checks.py has no check_prediction_length."""
        words = len(re.findall(r"[A-Za-z']+", str(text)))
        if words < minimum:
            return False, (f"Only {words} words selected. Please select at least {minimum} words "
                           "(a full article works best) for a reliable result.")
        return True, ""
from source_check import verify_source
from signals import assess_signals

MODEL_DIR = "saved_models"
MLFLOW_URI = "sqlite:///mlflow.db"
MLFLOW_EXPERIMENT = "fake-news-detection"
PLOTS_DIR = "plots"
MAX_WORDS = 10000      # LSTM vocabulary size
MAX_LEN = 300          # LSTM sequence length (longer = sees more of the article)
MIN_WORDS_FOR_RELIABLE = 40

# ---- Highlighting settings (sentences are far shorter than training articles,
# so the bar is deliberately high; treat highlights as hints, not verdicts)
HIGHLIGHT_THRESHOLD = 0.80
MIN_WORDS_PER_SENTENCE = 10
MAX_SENTENCES = 300

ML_MODEL_NAMES = ["Logistic Regression", "Naive Bayes", "Linear SVM",
                  "SGD Classifier", "Ridge"]


# ==============================
# Text preprocessing (shared by training and prediction)
# ==============================
# Source artifacts that let a model "cheat" (they mark the outlet, not the style).
ARTIFACT_PATTERNS = [
    r"\([^)]*corrected[^)]*\)",                              # "(This ... has been corrected ...)"
    r"\b[A-Z][A-Za-z.,'/ -]{1,40}\(\s*Reuters\s*\)\s*[-–—]*",  # "NAIROBI (Reuters) -"
    r"(?i)\b(?:on )?(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    r"(?i)\bimage via[^.\n]*",
    r"https?://\S+|www\.\S+", r"pic\.twitter\.com/\S+", r"@\w+",
    r"(?i)featured image[^.\n]*", r"(?i)\b(?:getty images?|screen ?grab|file photo)\b",
    r"(?i)\breuters\b", r"(?i)\b21st century wire\b",
]

def clean_text(text):
    text = str(text)
    for p in ARTIFACT_PATTERNS:
        text = re.sub(p, " ", text)
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def to_seq(tokenizer, texts):
    """The ONE place sequences are built. Used in training AND prediction, so the
    LSTM always sees the same format. Post-padding is correct because the
    Embedding layer uses mask_zero=True (padding is ignored)."""
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    return pad_sequences(tokenizer.texts_to_sequences(texts),
                         maxlen=MAX_LEN, padding="post", truncating="post")


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


# ==============================
# TRAINING
# ==============================
def train(csv_path, show_plots=True):
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run():
        _train(csv_path, show_plots)
        print("\nExperiment logged. To view it run:  mlflow ui --backend-store-uri " + MLFLOW_URI)


def _train(csv_path, show_plots=True):
    import matplotlib
    if not show_plots:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    import nltk
    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, classification_report, brier_score_loss, log_loss,
    )
    from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.optimizers import Adam

    nltk.download('stopwords', quiet=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    tf.keras.utils.set_random_seed(42)

    TEST_SIZE, CALIBRATION_SIZE, MAX_DF = 0.2, 0.25, 0.7
    LSTM_EPOCHS, LSTM_BATCH = 10, 64          # early stopping picks the best epoch
    LR_MAX_ITER = 1000
    mlflow.log_params({
        "test_size": TEST_SIZE, "calibration_fraction": CALIBRATION_SIZE,
        "tfidf_max_df": MAX_DF, "lstm_max_epochs": LSTM_EPOCHS,
        "lstm_batch_size": LSTM_BATCH, "lstm_vocab_size": MAX_WORDS,
        "lstm_sequence_length": MAX_LEN, "lstm_type": "BiLSTM(64)+mask_zero",
        "calibration_method": "sigmoid", "dataset_file": os.path.basename(csv_path),
    })

    def save_confusion(y_true, y_pred, name):
        plt.figure()
        sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix: {name}')
        plt.xlabel('Predicted'); plt.ylabel('Actual')
        path = os.path.join(PLOTS_DIR, "confusion_" + name.replace(" ", "_") + ".png")
        plt.savefig(path, bbox_inches="tight")
        mlflow.log_artifact(path, artifact_path="confusion_matrices")
        if show_plots:
            plt.show()
        plt.close()

    def report(name, y_true, fake_probs, probs2d=None):
        y_pred = (fake_probs >= 0.5).astype(int)
        acc = accuracy_score(y_true, y_pred)
        brier = brier_score_loss(y_true, fake_probs)
        ll = log_loss(y_true, np.clip(fake_probs, 1e-6, 1 - 1e-6), labels=[0, 1])
        print(f"\n{name} Accuracy: {acc:.4f}   Brier: {brier:.4f}   LogLoss: {ll:.4f}")
        print(classification_report(y_true, y_pred, zero_division=0))
        key = name.lower().replace(" ", "_")
        mlflow.log_metrics({
            key + "_accuracy": acc,
            key + "_precision": precision_score(y_true, y_pred, zero_division=0),
            key + "_recall": recall_score(y_true, y_pred, zero_division=0),
            key + "_f1": f1_score(y_true, y_pred, zero_division=0),
            key + "_brier_score": brier, key + "_log_loss": ll,
        })
        save_confusion(y_true, y_pred, name)
        return acc

    # ---- Load dataset, clean, DEDUPLICATE (duplicates leak between train/test)
    df = pd.read_csv(csv_path)
    df = df[['text', 'label']].dropna().copy()
    df['text'] = df['text'].apply(clean_text)
    df = df[df['text'].str.len() > 0]
    before = len(df)
    df = df.drop_duplicates(subset='text')
    print(f"Removed {before - len(df)} duplicate articles ({len(df)} remain)")
    mlflow.log_param("dataset_rows", len(df))

    X, y = df['text'], df['label'].astype(int)
    if set(y.unique()) != {0, 1}:
        raise ValueError("Labels must be 0 = Real and 1 = Fake, and both must be present.")

    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=42, stratify=y)
    X_train, X_calib, y_train, y_calib = train_test_split(
        X_dev, y_dev, test_size=CALIBRATION_SIZE, random_state=43, stratify=y_dev)
    print(f"Data split: train={len(X_train)}, calibration={len(X_calib)}, test={len(X_test)}")

    # ---- Classical models (calibrated)
    tfidf = TfidfVectorizer(stop_words='english', max_df=MAX_DF)
    X_train_t = tfidf.fit_transform(X_train)
    X_calib_t = tfidf.transform(X_calib)
    X_test_t = tfidf.transform(X_test)

    base_models = {
        "Logistic Regression": LogisticRegression(max_iter=LR_MAX_ITER),
        "Naive Bayes": MultinomialNB(),
        "Linear SVM": LinearSVC(random_state=42),
        "SGD Classifier": SGDClassifier(loss="modified_huber", max_iter=1000,
                                        tol=1e-3, random_state=42),
        "Ridge": RidgeClassifier(),
    }
    models = {}
    for name, base in base_models.items():
        try:
            m = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3)
        except TypeError:   # older scikit-learn
            m = CalibratedClassifierCV(base_estimator=base, method="sigmoid", cv=3)
        m.fit(X_train_t, y_train)
        models[name] = m

    calib_probs, test_probs, results = {}, {}, {}
    for name, m in models.items():
        fi = list(m.classes_).index(1)
        calib_probs[name] = m.predict_proba(X_calib_t)[:, fi]
        test_probs[name] = m.predict_proba(X_test_t)[:, fi]
        results[name] = report(name, y_test, test_probs[name])

    # ---- LSTM: BiLSTM + masking + early stopping + gradient clipping
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)
    X_train_pad = to_seq(tokenizer, X_train)
    X_calib_pad = to_seq(tokenizer, X_calib)
    X_test_pad = to_seq(tokenizer, X_test)

    lstm_model = Sequential([
        Embedding(input_dim=MAX_WORDS, output_dim=128, mask_zero=True),
        Bidirectional(LSTM(64, dropout=0.2)),
        Dropout(0.3),
        Dense(1, activation='sigmoid'),
    ])
    lstm_model.compile(loss='binary_crossentropy',
                       optimizer=Adam(learning_rate=1e-3, clipnorm=1.0),
                       metrics=['accuracy'])
    history = lstm_model.fit(
        X_train_pad, y_train.values, epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH,
        validation_split=0.1,
        callbacks=[EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)],
    )

    # Calibrate the LSTM on held-out data (on its logit, not the squashed output)
    lstm_calib_raw = lstm_model.predict(X_calib_pad, verbose=0).reshape(-1)
    lstm_calibrator = LogisticRegression(max_iter=LR_MAX_ITER)
    lstm_calibrator.fit(_logit(lstm_calib_raw), y_calib)
    fi = list(lstm_calibrator.classes_).index(1)
    calib_probs["LSTM"] = lstm_calibrator.predict_proba(_logit(lstm_calib_raw))[:, fi]
    lstm_test_raw = lstm_model.predict(X_test_pad, verbose=0).reshape(-1)
    test_probs["LSTM"] = lstm_calibrator.predict_proba(_logit(lstm_test_raw))[:, fi]
    results["LSTM"] = report("LSTM", y_test, test_probs["LSTM"])

    for epoch, (tl, vl, ta, va) in enumerate(zip(
            history.history['loss'], history.history['val_loss'],
            history.history['accuracy'], history.history['val_accuracy'])):
        mlflow.log_metrics({"lstm_train_loss": tl, "lstm_val_loss": vl,
                            "lstm_train_acc": ta, "lstm_val_acc": va}, step=epoch)

    # ---- Ensemble weights from the CALIBRATION set (never the test set):
    # a model with lower log loss gets more say, so a weak model can't drag the average down.
    weights = {}
    for name, p in calib_probs.items():
        ll = log_loss(y_calib, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])
        weights[name] = 1.0 / (ll + 0.05)
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}
    print("\nEnsemble weights:", {k: round(v, 3) for k, v in weights.items()})
    ens = sum(weights[n] * test_probs[n] for n in test_probs)
    results["Ensemble"] = report("Ensemble", y_test, ens)
    mlflow.log_params({"weight_" + k.replace(" ", "_"): round(v, 4) for k, v in weights.items()})

    if show_plots:
        plt.figure(figsize=(9, 5))
        sns.barplot(x=list(results.keys()), y=list(results.values()))
        plt.title('Model Accuracy Comparison'); plt.ylim(0, 1)
        plt.xticks(rotation=20); plt.show()
        plt.plot(history.history['accuracy'], label='Train'); plt.plot(history.history['val_accuracy'], label='Validation')
        plt.legend(); plt.title('LSTM Accuracy'); plt.show()
        plt.plot(history.history['loss'], label='Train'); plt.plot(history.history['val_loss'], label='Validation')
        plt.legend(); plt.title('LSTM Loss'); plt.show()

    # ---- Save everything (tokenizer as JSON: no pickle)
    joblib.dump(tfidf, os.path.join(MODEL_DIR, "tfidf.joblib"))
    for name, m in models.items():
        joblib.dump(m, os.path.join(MODEL_DIR, name.replace(" ", "_") + ".joblib"), compress=3)
    lstm_model.save(os.path.join(MODEL_DIR, "lstm.keras"))
    joblib.dump(lstm_calibrator, os.path.join(MODEL_DIR, "lstm_calibrator.joblib"), compress=3)
    with open(os.path.join(MODEL_DIR, "tokenizer.json"), "w", encoding="utf-8") as f:
        f.write(tokenizer.to_json())
    with open(os.path.join(MODEL_DIR, "weights.json"), "w") as f:
        json.dump(weights, f)
    with open(os.path.join(MODEL_DIR, "accuracies.json"), "w") as f:
        json.dump(results, f)
    print(f"\nModels saved to ./{MODEL_DIR}/  -> now run: python FakeNewsDetection.py serve")


# ==============================
# PREDICTION
# ==============================
_artifacts = None

def load_artifacts():
    global _artifacts
    if _artifacts is not None:
        return _artifacts
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.text import tokenizer_from_json

    needed = ["tfidf.joblib", "tokenizer.json", "weights.json", "lstm.keras", "lstm_calibrator.joblib"]
    if not all(os.path.exists(os.path.join(MODEL_DIR, f)) for f in needed):
        sys.exit("Saved models missing or from an older version. Run:  "
                 "python FakeNewsDetection.py train --csv FakeNews.csv")

    ml_models = {n: joblib.load(os.path.join(MODEL_DIR, n.replace(" ", "_") + ".joblib"))
                 for n in ML_MODEL_NAMES}
    with open(os.path.join(MODEL_DIR, "tokenizer.json"), encoding="utf-8") as f:
        tokenizer = tokenizer_from_json(f.read())
    with open(os.path.join(MODEL_DIR, "weights.json")) as f:
        weights = json.load(f)

    _artifacts = {
        "tfidf": joblib.load(os.path.join(MODEL_DIR, "tfidf.joblib")),
        "ml": ml_models,
        "tokenizer": tokenizer,
        "lstm": load_model(os.path.join(MODEL_DIR, "lstm.keras")),
        "lstm_calibrator": joblib.load(os.path.join(MODEL_DIR, "lstm_calibrator.joblib")),
        "weights": weights,
    }
    return _artifacts


def score_texts(texts, art):
    """Score a list of texts. Returns (per_model dict of arrays, weighted ensemble array).
    Used for BOTH the whole selection and individual sentences."""
    cleaned = [clean_text(t) for t in texts]
    vec = art["tfidf"].transform(cleaned)
    per_model = {}
    for name, m in art["ml"].items():
        fi = list(m.classes_).index(1)
        per_model[name] = m.predict_proba(vec)[:, fi]

    raw = art["lstm"].predict(to_seq(art["tokenizer"], cleaned), verbose=0).reshape(-1)
    cal = art["lstm_calibrator"]
    fi = list(cal.classes_).index(1)
    per_model["LSTM"] = cal.predict_proba(_logit(raw))[:, fi]

    names = list(per_model.keys())
    w = np.array([art["weights"].get(n, 1.0 / len(names)) for n in names])
    ensemble = np.average(np.vstack([per_model[n] for n in names]), axis=0, weights=w)
    return per_model, ensemble


def find_suspicious_sentences(text, art):
    """Score each sentence; return flagged sentences and an escaped HTML version."""
    spans = []
    for m in re.finditer(r'[^.!?\n]+[.!?]*', text):
        raw = m.group()
        stripped = raw.strip()
        if not stripped:
            continue
        start = m.start() + (len(raw) - len(raw.lstrip()))
        spans.append((start, start + len(stripped), stripped))
    spans = spans[:MAX_SENTENCES]

    scorable = [i for i, (_, _, s) in enumerate(spans)
                if len(clean_text(s).split()) >= MIN_WORDS_PER_SENTENCE]
    scores = {}
    if scorable:
        _, ens = score_texts([spans[i][2] for i in scorable], art)
        scores = {idx: float(p) for idx, p in zip(scorable, ens)}

    suspicious, flagged = [], set()
    for i, p in scores.items():
        if p >= HIGHLIGHT_THRESHOLD:
            start, end, s = spans[i]
            suspicious.append({"text": s, "start": start, "end": end,
                               "fake_percent": round(p * 100, 1)})
            flagged.add(i)

    def esc(t):
        return html.escape(t).replace("\n", "<br>")

    parts, cursor = [], 0
    for i, (start, end, s) in enumerate(spans):
        parts.append(esc(text[cursor:start]))
        if i in flagged:
            pct = round(scores[i] * 100, 1)
            parts.append(
                f'<mark class="fn-suspicious" title="Style resembles fake-news examples ({pct}%)" '
                f'style="background:#ffd54f;color:#000;border-radius:3px;padding:0 2px;">'
                f'{esc(s)}</mark>')
        else:
            parts.append(esc(s))
        cursor = end
    parts.append(esc(text[cursor:]))
    return suspicious, "".join(parts)


def predict_fakeness(text, page_host=None, page_protocol=None):
    art = load_artifacts()
    word_count = len(clean_text(text).split())

    per_model, ens = score_texts([text], art)
    style_percent = round(float(ens[0]) * 100, 1)       # signal 1: writing style only

    source = verify_source(page_host, page_protocol)     # signal 2: website
    assessment = assess_signals(style_percent, word_count, source)

    warning = None
    if word_count < MIN_WORDS_FOR_RELIABLE:
        warning = (f"Only {word_count} words selected. The models were trained on full articles, "
                   "so short text gives low-confidence results. Select more of the article.")

    suspicious_sentences, highlighted_html = find_suspicious_sentences(str(text), art)

    return {
        "verdict": assessment["verdict"],
        "verdict_level": assessment["verdict_level"],
        "agreement": assessment["agreement"],
        "style_percent": style_percent,
        "style_stance": assessment["style_stance"],
        "style_label": assessment["style_label"],
        "source": source,
        "word_count": word_count,
        "warning": warning,
        "models": {k: round(float(v[0]) * 100, 1) for k, v in per_model.items()},
        "suspicious_sentences": suspicious_sentences,
        "highlighted_html": highlighted_html,
    }


# ==============================
# API SERVER (used by the browser extension)
# ==============================
def serve(host="127.0.0.1", port=5000):
    from flask import Flask, request, jsonify

    app = Flask(__name__)
    load_artifacts()

    @app.after_request
    def add_cors(resp):
        # Only browser extensions may call this local API (not arbitrary websites).
        origin = request.headers.get("Origin", "")
        if origin.startswith(("chrome-extension://", "moz-extension://")):
            resp.headers["Access-Control-Allow-Origin"] = origin
            resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
            resp.headers["Vary"] = "Origin"
        return resp

    @app.route("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.route("/predict", methods=["POST", "OPTIONS"])
    def predict():
        if request.method == "OPTIONS":
            return ("", 204)
        data = request.get_json(silent=True) or {}
        text = (data.get("text") or "").strip()

        ok, message = check_text_usable(text)
        if not ok:
            return jsonify({"error": message}), 400

        # Require enough words for a meaningful score. "Scan anyway" does not bypass this.
        ok, message = check_prediction_length(text)
        if not ok:
            word_count = len(re.findall(r"[A-Za-z']+", text))
            return jsonify({"error": message, "word_count": word_count}), 400

        if not data.get("force"):
            nl = news_likeness(text)
            if not nl["is_news"]:
                return jsonify({
                    "not_news": True,
                    "message": "This doesn't look like a news article, so a fake/real score "
                               "would not be meaningful.",
                    "reasons": nl["reasons"],
                })

        try:
            return jsonify(predict_fakeness(text[:MAX_CHARS],
                                            page_host=data.get("host"),
                                            page_protocol=data.get("protocol")))
        except Exception:
            traceback.print_exc()
            return jsonify({"error": "The server could not analyse this text. "
                                     "Check the server terminal for the error."}), 500

    print(f"Fake news API running on http://{host}:{port}")
    app.run(host=host, port=port)


# ==============================
# CLI
# ==============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--csv", default="FakeNews.csv")
    p_train.add_argument("--no-plots", action="store_true")

    p_serve = sub.add_parser("serve")
    p_serve.add_argument("--port", type=int, default=5000)

    p_pred = sub.add_parser("predict")
    p_pred.add_argument("--text", required=True)
    p_pred.add_argument("--url", default=None, help="page URL the text came from (optional)")

    args = parser.parse_args()
    if args.cmd == "train":
        train(args.csv, show_plots=not args.no_plots)
    elif args.cmd == "serve":
        serve(port=args.port)
    elif args.cmd == "predict":
        for check in (check_text_usable, check_prediction_length):
            ok, message = check(args.text)
            if not ok:
                print(json.dumps({"error": message}, indent=2))
                sys.exit(1)
        host = protocol = None
        if args.url:
            u = urlparse(args.url if "://" in args.url else "https://" + args.url)
            host, protocol = u.hostname, u.scheme + ":"
        print(json.dumps(predict_fakeness(args.text[:MAX_CHARS], page_host=host, page_protocol=protocol), indent=2))
