"""
Breast tumour classification (malignant vs benign): a careful model evaluation.

Data: Wisconsin Diagnostic Breast Cancer dataset, bundled with scikit-learn (no download).
569 tumours, 30 measurements of cell nuclei from a biopsy image (size, shape, texture).

What this project demonstrates:
    - an honest evaluation: a test set that is touched only once, cross-validation for model choice
    - a decision threshold chosen for a business reason (missing a cancer is worse than a false alarm)
    - confidence intervals on the final result (the test set is small)
    - model checks: feature importance on correlated features, probability calibration

This is a learning project, not a medical device.

Usage:  python breast_cancer_analysis.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict,
                                     cross_validate, train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
IMAGES = BASE / "images"
SEED = 42
TARGET_RECALL = 0.98   # we want to catch at least 98% of malignant tumours

BLUE, ORANGE, INK, MUTED, GRID, SURFACE = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
GREYBLUE = "#a9c4ea"


def style(ax, grid="y"):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c9c8c3")
    ax.tick_params(colors=MUTED, length=0)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def title(ax, text):
    ax.set_title(text, loc="left", color=INK, fontsize=12, fontweight="bold")


def wilson(k, n, z=1.96):
    """95% Wilson confidence interval for a proportion k / n."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def get_models():
    return {
        "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7)),
        "Random forest": RandomForestClassifier(n_estimators=300, random_state=SEED),
        "Gradient boosting": HistGradientBoostingClassifier(random_state=SEED),
    }


def main():
    RESULTS.mkdir(exist_ok=True)
    IMAGES.mkdir(exist_ok=True)

    # ------------------------------------------------ data
    raw = load_breast_cancer(as_frame=True)
    X = raw.data
    y = (raw.target == 0).astype(int)   # in scikit-learn 0 = malignant; we make malignant = 1
    n_mal = int(y.sum())
    print(f"Tumours: {len(y)} | malignant: {n_mal} ({n_mal/len(y):.1%}) | benign: {len(y)-n_mal} | features: {X.shape[1]}")
    print(f"Baseline: always predicting 'benign' gives accuracy {1 - n_mal/len(y):.1%} but finds 0% of cancers")

    # which measurements separate the classes best (Cohen's d)
    mal, ben = X[y == 1], X[y == 0]
    pooled = np.sqrt(((len(mal) - 1) * mal.var() + (len(ben) - 1) * ben.var()) / (len(X) - 2))
    d = ((mal.mean() - ben.mean()) / pooled).sort_values(key=abs, ascending=False)
    d.round(2).to_csv(RESULTS / "feature_effect_sizes.csv", header=["cohens_d"])
    print("\nTop measurements by difference between malignant and benign (Cohen's d):")
    print(d.head(5).round(2).to_string())

    corr = X.corr().abs()
    pairs = (corr.where(np.triu(np.ones(corr.shape), 1).astype(bool)).stack())
    print(f"Highly correlated feature pairs (|r| > 0.9): {(pairs > 0.9).sum()} of {len(pairs)}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    print(f"\nTrain: {len(X_train)} | test: {len(X_test)} (test set is used only once, at the end)")

    # ------------------------------------------------ model comparison with repeated CV
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)
    rows = []
    for name, model in get_models().items():
        r = cross_validate(model, X_train, y_train, cv=cv, scoring=["roc_auc", "recall", "precision"])
        rows.append({"model": name,
                     "roc_auc": r["test_roc_auc"].mean(), "roc_auc_sd": r["test_roc_auc"].std(),
                     "recall": r["test_recall"].mean(), "recall_sd": r["test_recall"].std(),
                     "precision": r["test_precision"].mean(), "precision_sd": r["test_precision"].std()})
    comp = pd.DataFrame(rows).set_index("model")
    comp.round(4).to_csv(RESULTS / "model_comparison_cv.csv")
    print("\nMODEL COMPARISON (5-fold CV repeated 5 times, on the training set; threshold 0.5)")
    print(comp[["roc_auc", "roc_auc_sd", "recall", "precision"]].round(3).to_string())

    best_name = comp["roc_auc"].idxmax()
    best = get_models()[best_name]
    print(f"\nChosen model (highest CV ROC-AUC): {best_name}")

    # ------------------------------------------------ threshold chosen on out-of-fold predictions
    oof = cross_val_predict(best, X_train, y_train, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                            method="predict_proba")[:, 1]
    ths = np.linspace(0.01, 0.99, 99)
    rec = np.array([recall_score(y_train, oof >= t) for t in ths])
    prec = np.array([precision_score(y_train, oof >= t, zero_division=0) for t in ths])
    ok = np.where(rec >= TARGET_RECALL)[0]
    idx = ok[-1] if len(ok) else int(np.argmax(rec))
    thr = float(ths[idx])
    print(f"\nTHRESHOLD (chosen on out-of-fold predictions, target recall >= {TARGET_RECALL:.0%}): {thr:.2f}")
    print(f"  at this threshold on training folds: recall {rec[idx]:.3f}, precision {prec[idx]:.3f}")

    # ------------------------------------------------ final test (one time)
    best.fit(X_train, y_train)
    p_test = best.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, p_test)
    out = []
    for label, t in (("default 0.5", 0.5), (f"chosen {thr:.2f}", thr)):
        pred = (p_test >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
        r_lo, r_hi = wilson(tp, tp + fn)
        p_lo, p_hi = wilson(tp, tp + fp)
        out.append({"threshold": label, "TP": tp, "FN": fn, "FP": fp, "TN": tn,
                    "recall": tp / (tp + fn), "recall_ci_low": r_lo, "recall_ci_high": r_hi,
                    "precision": tp / (tp + fp) if tp + fp else np.nan, "precision_ci_low": p_lo, "precision_ci_high": p_hi})
    test = pd.DataFrame(out).set_index("threshold")
    test.round(4).to_csv(RESULTS / "test_results.csv")
    print(f"\nTEST SET ({len(y_test)} tumours, {int(y_test.sum())} malignant) | ROC-AUC {auc:.3f}")
    print(test[["TP", "FN", "FP", "TN", "recall", "recall_ci_low", "recall_ci_high", "precision"]].round(3).to_string())

    # ------------------------------------------------ permutation importance
    pi = permutation_importance(best, X_test, y_test, scoring="roc_auc", n_repeats=30, random_state=SEED)
    imp = pd.Series(pi.importances_mean, index=X.columns).sort_values(ascending=False)
    imp.round(4).to_csv(RESULTS / "permutation_importance.csv", header=["auc_drop"])
    print("\nPERMUTATION IMPORTANCE (drop in test ROC-AUC when a feature is shuffled)")
    print(imp.head(5).round(4).to_string())
    print("Note: many features are strongly correlated, so importance is spread among them.")

    # ------------------------------------------------ charts
    # 01 effect sizes
    top = d.head(10)[::-1]
    fig, ax = plt.subplots(figsize=(8, 4.6), facecolor=SURFACE)
    style(ax, grid="x")
    ax.barh(top.index, top.abs().values, color=BLUE, height=0.62)
    for i, v in enumerate(top.abs().values):
        ax.text(v + 0.03, i, f"{v:.1f}", va="center", color=INK, fontsize=9)
    ax.set_xticks([])
    title(ax, "Measurements that differ most between malignant and benign (Cohen's d)")
    fig.tight_layout(); fig.savefig(IMAGES / "01_feature_differences.png", dpi=150, facecolor=SURFACE); plt.close(fig)

    # 02 model comparison
    fig, ax = plt.subplots(figsize=(8, 3.8), facecolor=SURFACE)
    style(ax, grid="x")
    order = comp.sort_values("roc_auc")
    cols = [BLUE if m == best_name else GREYBLUE for m in order.index]
    ax.barh(order.index, order["roc_auc"], xerr=order["roc_auc_sd"], color=cols, height=0.6,
            error_kw={"ecolor": MUTED, "elinewidth": 1})
    ax.set_xlim(order["roc_auc"].min() - 0.03, 1.0)
    left = order["roc_auc"].min() - 0.03
    for i, v in enumerate(order["roc_auc"]):
        ax.text(left + 0.002, i, f"{v:.3f}", va="center", ha="left", color="white" if v == order["roc_auc"].max() else INK,
                fontsize=10, fontweight="bold")
    ax.set_xlabel("Cross-validated ROC-AUC (bar = mean, whisker = standard deviation)", color=MUTED)
    title(ax, "Model comparison on the training set")
    fig.tight_layout(); fig.savefig(IMAGES / "02_model_comparison.png", dpi=150, facecolor=SURFACE); plt.close(fig)

    # 03 threshold trade-off
    fig, ax = plt.subplots(figsize=(8, 4), facecolor=SURFACE)
    style(ax)
    ax.plot(ths, rec, color=BLUE, linewidth=2.25, label="Recall (cancers found)")
    ax.plot(ths, prec, color=ORANGE, linewidth=2.25, label="Precision (alarms that are real)")
    ax.axvline(thr, color=MUTED, linestyle="--", linewidth=1)
    ax.text(thr + 0.01, 0.45, f"chosen threshold {thr:.2f}", color=INK, fontsize=9)
    ax.set_xlabel("Probability threshold for 'malignant'", color=MUTED)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, labelcolor=INK, loc="lower center")
    title(ax, "Threshold trade-off (out-of-fold predictions on the training set)")
    fig.tight_layout(); fig.savefig(IMAGES / "03_threshold_tradeoff.png", dpi=150, facecolor=SURFACE); plt.close(fig)

    # 04 confusion matrix (chosen threshold)
    row = test.iloc[1]
    cm = np.array([[row["TN"], row["FP"]], [row["FN"], row["TP"]]]).astype(int)
    fig, ax = plt.subplots(figsize=(4.8, 4.2), facecolor=SURFACE)
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", fontsize=18, color=INK if v < cm.max() * 0.6 else "white")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted benign", "Predicted malignant"], color=MUTED)
    ax.set_yticklabels(["Actually benign", "Actually malignant"], color=MUTED)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    title(ax, f"Test set, threshold {thr:.2f}")
    fig.tight_layout(); fig.savefig(IMAGES / "04_confusion_matrix.png", dpi=150, facecolor=SURFACE); plt.close(fig)

    # 05 permutation importance
    t10 = imp.head(8)[::-1]
    fig, ax = plt.subplots(figsize=(8, 4), facecolor=SURFACE)
    style(ax, grid="x")
    ax.barh(t10.index, t10.values, color=BLUE, height=0.62)
    ax.set_xlabel("Drop in test ROC-AUC when the feature is shuffled", color=MUTED)
    title(ax, "Permutation importance (correlated features share the credit)")
    fig.tight_layout(); fig.savefig(IMAGES / "05_permutation_importance.png", dpi=150, facecolor=SURFACE); plt.close(fig)

    # 06 calibration
    frac, mean_pred = calibration_curve(y_train, oof, n_bins=8, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 4.4), facecolor=SURFACE)
    style(ax)
    ax.plot([0, 1], [0, 1], color=MUTED, linestyle="--", linewidth=1)
    ax.plot(mean_pred, frac, color=BLUE, linewidth=2.25, marker="o", markersize=5)
    ax.set_xlabel("Predicted probability of malignant", color=MUTED)
    ax.set_ylabel("Share that really were malignant", color=MUTED)
    title(ax, "Calibration (out-of-fold)")
    fig.tight_layout(); fig.savefig(IMAGES / "06_calibration.png", dpi=150, facecolor=SURFACE); plt.close(fig)

    (RESULTS / "findings.txt").write_text(
        f"Model: {best_name}; CV ROC-AUC {comp.loc[best_name, 'roc_auc']:.3f}; threshold {thr:.2f}\n"
        f"Test: recall {row['recall']:.3f} (95% CI {row['recall_ci_low']:.3f}-{row['recall_ci_high']:.3f}), "
        f"missed {int(row['FN'])} cancers, false alarms {int(row['FP'])}\n", encoding="utf-8")
    print("\nDone. Tables are in results/, charts are in images/.")


if __name__ == "__main__":
    main()
