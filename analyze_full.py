"""
Analyze human ratings for the 3-model x 2-domain x 2-condition x 20-item
"lost in conversation" creative writing study (240 stories total).

Expects three (or more) rater CSVs, each with columns:
    story_uid, title, story_text, relevance, coherence, empathy,
    surprise, engagement, complexity
(produced by make_templates.py; score columns filled in with 1-5 by raters)

story_uid format: "{domain}_{item_id}_{condition}_{model}"

Usage:
    python analyze_full.py rater1_blinded.csv rater2_blinded.csv rater3_blinded.csv
"""
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from itertools import combinations
import statsmodels.formula.api as smf

CRITERIA = ["relevance", "coherence", "empathy", "surprise", "engagement", "complexity"]


def load_ratings(paths):
    """Return dict: criterion -> wide DataFrame (rows=story_uid, cols=rater1..raterK)."""
    per_rater = []
    for p in paths:
        df = pd.read_csv(p).set_index("story_uid")
        per_rater.append(df)

    wide_by_criterion = {}
    for crit in CRITERIA:
        cols = {}
        for i, df in enumerate(per_rater, start=1):
            cols[f"rater{i}"] = df[crit]
        wide = pd.DataFrame(cols)
        before = len(wide)
        wide = wide.dropna()
        after = len(wide)
        if after < before:
            print(f"  [{crit}] dropped {before - after} stories with missing ratings")
        wide_by_criterion[crit] = wide.astype(int)
    return wide_by_criterion


def fleiss_kappa(ratings_wide, categories=(1, 2, 3, 4, 5)):
    n_items, n_raters = ratings_wide.shape
    cat_index = {c: i for i, c in enumerate(categories)}
    counts = np.zeros((n_items, len(categories)))
    for i, (_, row) in enumerate(ratings_wide.iterrows()):
        for v in row.values:
            counts[i, cat_index[v]] += 1
    p_j = counts.sum(axis=0) / (n_items * n_raters)
    P_i = (np.sum(counts ** 2, axis=1) - n_raters) / (n_raters * (n_raters - 1))
    P_bar = P_i.mean()
    P_e = np.sum(p_j ** 2)
    return (P_bar - P_e) / (1 - P_e)


def avg_pairwise_weighted_kappa(ratings_wide, weights="quadratic"):
    raters = ratings_wide.columns.tolist()
    scores = [cohen_kappa_score(ratings_wide[r1], ratings_wide[r2], weights=weights)
              for r1, r2 in combinations(raters, 2)]
    return np.mean(scores)


def icc_2k(ratings_wide):
    data = ratings_wide.values.astype(float)
    n, k = data.shape
    mean_items = data.mean(axis=1)
    mean_raters = data.mean(axis=0)
    grand_mean = data.mean()
    SSR = k * np.sum((mean_items - grand_mean) ** 2)
    SSC = n * np.sum((mean_raters - grand_mean) ** 2)
    SSE = np.sum((data - mean_items[:, None] - mean_raters[None, :] + grand_mean) ** 2)
    MSR = SSR / (n - 1)
    MSC = SSC / (k - 1)
    MSE = SSE / ((n - 1) * (k - 1))
    return (MSR - MSE) / (MSR + (MSC - MSE) / n)


def uid_to_meta(uid):
    domain, item_id, condition, model = uid.rsplit("_", 3)
    return domain, item_id, condition, model


def build_long_df(wide_by_criterion):
    """One row per (story, criterion) with mean rater score + parsed metadata."""
    records = []
    for crit, wide in wide_by_criterion.items():
        mean_score = wide.mean(axis=1)
        for uid, score in mean_score.items():
            domain, item_id, condition, model = uid_to_meta(uid)
            records.append({
                "story_uid": uid, "criterion": crit, "score": score,
                "domain": domain, "item_id": item_id,
                "condition": condition, "model": model,
            })
    return pd.DataFrame(records)


def run_mixed_model(long_df, criterion):
    """score ~ condition * model * domain, random intercept per item_id."""
    sub = long_df[long_df["criterion"] == criterion].copy()
    sub["item_key"] = sub["domain"] + "_" + sub["item_id"]  # random-effect grouping
    sub["condition"] = pd.Categorical(sub["condition"], categories=["full", "sharded"])
    try:
        md = smf.mixedlm(
            "score ~ C(condition, Treatment('full')) * C(model) * C(domain)",
            sub, groups=sub["item_key"],
        )
        result = md.fit(reml=True)
        return result
    except Exception as e:
        print(f"  Mixed model failed for {criterion}: {e}")
        return None


def summarize_condition_effect(long_df):
    print("\n=== Mean score by condition x model (all criteria combined) ===")
    print(long_df.groupby(["model", "condition"])["score"].agg(["mean", "std", "count"]))

    print("\n=== Mean score by criterion x condition ===")
    print(long_df.groupby(["criterion", "condition"])["score"].mean().unstack())


def main(paths):
    print(f"Loading ratings from {len(paths)} raters...\n")
    wide_by_criterion = load_ratings(paths)

    print("=== Inter-rater agreement per criterion ===")
    for crit, wide in wide_by_criterion.items():
        fk = fleiss_kappa(wide)
        wk = avg_pairwise_weighted_kappa(wide)
        icc = icc_2k(wide)
        print(f"{crit:12s}  Fleiss k={fk:.3f}   weighted k={wk:.3f}   ICC(2,k)={icc:.3f}   n={len(wide)}")

    long_df = build_long_df(wide_by_criterion)
    summarize_condition_effect(long_df)

    print("\n=== Mixed-effects model: score ~ condition * model * domain (random intercept per item) ===")
    for crit in CRITERIA:
        print(f"\n--- {crit} ---")
        result = run_mixed_model(long_df, crit)
        if result is not None:
            # print just the condition-related rows for readability
            params = result.params
            pvals = result.pvalues
            for name in params.index:
                if "condition" in name.lower():
                    print(f"  {name}: coef={params[name]:.3f}, p={pvals[name]:.4f}")

    long_df.to_csv("long_format_scores.csv", index=False)
    print("\nSaved per-story, per-criterion mean scores to long_format_scores.csv")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1:])
