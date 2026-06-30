"""
FairFinance AI — Analysis Script with strict metrics (v2)
Uses the strict checklist-based accuracy scoring output as the source of truth.
This version does not require the fairlearn package at runtime.
Classifier section is treated as an auxiliary diagnostic for SES signal recoverability,
not direct proof of bias.
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, StratifiedKFold, permutation_test_score
from sklearn.preprocessing import LabelEncoder

PROMPT_TYPES = ["zero_shot", "few_shot", "fairness_instructed"]


def demographic_parity_difference_manual(y_pred, sensitive):
    sensitive = pd.Series(sensitive)
    y_pred = pd.Series(y_pred).astype(float)
    high_mask = sensitive == "high"
    low_mask = sensitive == "low"
    if high_mask.sum() == 0 or low_mask.sum() == 0:
        return None
    return float(y_pred[high_mask].mean() - y_pred[low_mask].mean())


def demographic_parity_ratio_manual(y_pred, sensitive):
    sensitive = pd.Series(sensitive)
    y_pred = pd.Series(y_pred).astype(float)
    high_mask = sensitive == "high"
    low_mask = sensitive == "low"
    if high_mask.sum() == 0 or low_mask.sum() == 0:
        return None
    high_rate = float(y_pred[high_mask].mean())
    low_rate = float(y_pred[low_mask].mean())
    if high_rate == 0:
        return None
    return float(low_rate / high_rate)


def train_ses_classifier(df_model: pd.DataFrame):
    """
    Diagnostic analysis only: test whether SES group is recoverable from response text.
    Returns CV accuracy, std, permutation p-value, and correctly aligned top features by class.
    """
    sub = df_model[df_model["ses_group"].isin(["high", "low"])].copy()
    if len(sub) < 10:
        return None

    vectorizer = TfidfVectorizer(max_features=1000, stop_words="english", ngram_range=(1, 2))
    X = vectorizer.fit_transform(sub["response"].fillna(""))
    le = LabelEncoder()
    y = le.fit_transform(sub["ses_group"])
    class_names = list(le.classes_)  # usually ['high', 'low']

    clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced", random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    clf.fit(X, y)

    # In binary logistic regression, positive coefficients correspond to class_names[1].
    feature_names = vectorizer.get_feature_names_out()
    coef = clf.coef_[0]
    sorted_idx = np.argsort(coef)
    top_for_class0 = [feature_names[i] for i in sorted_idx[:5]]
    top_for_class1 = [feature_names[i] for i in sorted_idx[-5:][::-1]]

    try:
        _, _, pvalue = permutation_test_score(
            clf, X, y, cv=cv, scoring="accuracy", n_permutations=200, n_jobs=1, random_state=42
        )
        pvalue = float(pvalue)
    except Exception:
        pvalue = None

    return {
        "cv_accuracy_mean": float(scores.mean()),
        "cv_accuracy_std": float(scores.std()),
        "p_value": pvalue,
        "class_0_name": class_names[0],
        "class_1_name": class_names[1],
        "top_class_0_words": top_for_class0,
        "top_class_1_words": top_for_class1,
    }


def load_strict_data(strict_input: str):
    df = pd.read_csv(strict_input)
    required = {"model", "prompt_type", "ses_group", "response", "accuracy_score", "quality_score", "word_count"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Strict input is missing required columns: {missing}")

    threshold = float(df["accuracy_score"].median())
    df["accuracy_binary"] = (df["accuracy_score"] >= threshold).astype(int)
    return df, threshold


def analyze(strict_input: str = "fairfinance_accuracy_strict.csv", output_file: str = "fairfinance_analyzed_strict.csv"):
    print(f"Loading strict scored dataset: {strict_input}")
    df, acc_threshold = load_strict_data(strict_input)
    print(f"  Total records: {len(df)}")
    print(f"  Accuracy threshold (median, for binary parity metrics): {acc_threshold:.4f}")

    df_hl = df[df["ses_group"].isin(["high", "low"])].copy()

    print("\n── Mean Scores by SES Group ─────────────────────────")
    for model in df["model"].dropna().unique():
        sub = df_hl[df_hl["model"] == model]
        print(f"\n  [{model}]")
        for group in ["high", "low"]:
            g = sub[sub["ses_group"] == group]
            print(
                f"    {group:>4} SES: quality={g['quality_score'].mean():.2f}  "
                f"accuracy={g['accuracy_score'].mean():.3f}  "
                f"raw_acc={g['accuracy_score_raw'].mean():.3f}  "
                f"penalty={g['penalty_score'].mean():.3f}  "
                f"words={g['word_count'].mean():.0f}"
            )

    print("\n── Accuracy Gap |E[A|high] - E[A|low]| ─────────────")
    for model in df["model"].dropna().unique():
        sub = df_hl[df_hl["model"] == model]
        high = sub[sub["ses_group"] == "high"]["accuracy_score"].mean()
        low = sub[sub["ses_group"] == "low"]["accuracy_score"].mean()
        print(f"  [{model}]  gap={abs(high - low):.4f}  (high={high:.3f}, low={low:.3f})")

    print("\n── Parity Metrics on Accuracy Binary ────────────────")
    print(f"  {'Model':<30} {'Prompt Type':<25} {'DPD':<8} {'DIR':<8}")
    print(f"  {'-'*30} {'-'*25} {'-'*8} {'-'*8}")

    results = []
    for model in df["model"].dropna().unique():
        for ptype in PROMPT_TYPES:
            sub = df_hl[(df_hl["model"] == model) & (df_hl["prompt_type"] == ptype)]
            if len(sub) < 4:
                continue

            y_pred = sub["accuracy_binary"].astype(int).values
            sensitive = sub["ses_group"].values
            dpd = demographic_parity_difference_manual(y_pred, sensitive)
            dir_ = demographic_parity_ratio_manual(y_pred, sensitive)
            dpd = round(float(dpd), 4) if dpd is not None else None
            dir_ = round(float(dir_), 4) if dir_ is not None else None

            results.append({
                "model": model,
                "prompt_type": ptype,
                "dpd": dpd,
                "dir": dir_,
                "metric_basis": "accuracy_binary",
                "accuracy_threshold": round(acc_threshold, 4),
            })
            print(f"  {model:<30} {ptype:<25} {str(dpd):<8} {str(dir_):<8}")

    print("\n── Accuracy-Fairness Tradeoff ───────────────────────")
    print(f"  {'Model':<30} {'Prompt Type':<25} {'Acc Gap':<10} {'Qual Gap':<10} {'Penalty Gap':<12}")
    print(f"  {'-'*30} {'-'*25} {'-'*10} {'-'*10} {'-'*12}")
    tradeoff_rows = []
    for model in df["model"].dropna().unique():
        for ptype in PROMPT_TYPES:
            sub = df_hl[(df_hl["model"] == model) & (df_hl["prompt_type"] == ptype)]
            if len(sub) < 4:
                continue
            high = sub[sub["ses_group"] == "high"]
            low = sub[sub["ses_group"] == "low"]
            acc_gap = round(float(high["accuracy_score"].mean() - low["accuracy_score"].mean()), 4)
            qual_gap = round(float(high["quality_score"].mean() - low["quality_score"].mean()), 4)
            penalty_gap = round(float(high["penalty_score"].mean() - low["penalty_score"].mean()), 4)
            print(f"  {model:<30} {ptype:<25} {acc_gap:<10} {qual_gap:<10} {penalty_gap:<12}")
            tradeoff_rows.append({
                "model": model,
                "prompt_type": ptype,
                "accuracy_gap": acc_gap,
                "quality_gap": qual_gap,
                "penalty_gap": penalty_gap,
                "high_accuracy": round(float(high["accuracy_score"].mean()), 4),
                "low_accuracy": round(float(low["accuracy_score"].mean()), 4),
                "high_quality": round(float(high["quality_score"].mean()), 4),
                "low_quality": round(float(low["quality_score"].mean()), 4),
            })

    print("\n── SES Signal Recoverability (Auxiliary Diagnostic) ───")
    classifier_rows = []
    for model in df["model"].dropna().unique():
        sub = df_hl[df_hl["model"] == model]
        clf_result = train_ses_classifier(sub)
        if clf_result is not None:
            class0 = clf_result["class_0_name"]
            class1 = clf_result["class_1_name"]
            print(f"\n  [{model}]")
            print(
                f"    CV accuracy: {clf_result['cv_accuracy_mean']:.3f} ± {clf_result['cv_accuracy_std']:.3f}  "
                f"(chance=0.500, permutation p={clf_result['p_value'] if clf_result['p_value'] is not None else 'NA'})"
            )
            print("    Interpretation: high accuracy means SES-related patterns are recoverable from text; it does not by itself prove unfairness.")
            print(f"    Top {class0.upper()}-associated words: {clf_result['top_class_0_words']}")
            print(f"    Top {class1.upper()}-associated words: {clf_result['top_class_1_words']}")
            classifier_rows.append({
                "model": model,
                "cv_accuracy_mean": round(clf_result['cv_accuracy_mean'], 4),
                "cv_accuracy_std": round(clf_result['cv_accuracy_std'], 4),
                "permutation_p_value": None if clf_result['p_value'] is None else round(clf_result['p_value'], 6),
                f"top_{class0}_words": "; ".join(clf_result['top_class_0_words']),
                f"top_{class1}_words": "; ".join(clf_result['top_class_1_words']),
                "interpretation": "SES-related linguistic patterns are recoverable from the generated text; this is diagnostic, not direct evidence of unfairness.",
            })

    metrics_df = pd.DataFrame(results)
    tradeoff_df = pd.DataFrame(tradeoff_rows)
    classifier_df = pd.DataFrame(classifier_rows)

    df.to_csv(output_file, index=False)
    metrics_df.to_csv("fairfinance_metrics_strict.csv", index=False)
    tradeoff_df.to_csv("fairfinance_tradeoff_strict_from_analysis.csv", index=False)
    classifier_df.to_csv("fairfinance_classifier_strict.csv", index=False)

    print(f"\n✅ Strict analyzed dataset saved to {output_file}")
    print("✅ Fairness metrics saved to fairfinance_metrics_strict.csv")
    print("✅ Accuracy-fairness tradeoff saved to fairfinance_tradeoff_strict_from_analysis.csv")
    print("✅ SES signal recoverability summary saved to fairfinance_classifier_strict.csv")
    return df, metrics_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict_input", default="fairfinance_accuracy_strict.csv")
    parser.add_argument("--output", default="fairfinance_analyzed_strict.csv")
    args = parser.parse_args()
    print("FairFinance AI — Analysis with strict metrics\n")
    analyze(args.strict_input, args.output)
