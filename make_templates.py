"""
Generate blinded annotation templates for human evaluation, covering all 6
HANNA criteria (Relevance, Coherence, Empathy, Surprise, Engagement,
Complexity), for a results.jsonl of ANY size / number of models.

Usage:
    python make_templates.py results.jsonl --n_raters 3 --outdir templates/


     python3 make_templates.py results_qwen.jsonl --n_raters 3 --outdir templates_qwen/

Produces:
    templates/master_annotation_template.csv   (unblinded, for your records)
    templates/rater1_blinded.csv ... rater{n}_blinded.csv
        each with columns: story_uid, title, story_text,
        relevance, coherence, empathy, surprise, engagement, complexity
        (score columns left empty for the rater to fill in, 1-5)

story_uid format: "{domain}_{item_id}_{condition}_{model}"
Rows are independently shuffled per rater file so raters can't infer the
condition/model pattern from ordering.
"""
import argparse
import json
import os
import random
import csv

CRITERIA = ["relevance", "coherence", "empathy", "surprise", "engagement", "complexity"]


def sanitize_model_name(m):
    return m.replace("/", "-").replace(":", "-")


def load_rows(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_path")
    ap.add_argument("--n_raters", type=int, default=3)
    ap.add_argument("--outdir", default="templates")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rows = load_rows(args.results_path)
    print(f"Loaded {len(rows)} stories from {args.results_path}")

    models = sorted(set(r["model"] for r in rows))
    domains = sorted(set(r["domain"] for r in rows))
    conditions = sorted(set(r["condition"] for r in rows))
    print(f"Models: {models}")
    print(f"Domains: {domains}")
    print(f"Conditions: {conditions}")
    print(f"Expected total rows = {len(models)} models x {len(domains)} domains x "
          f"{len(conditions)} conditions x N items")

    # master (unblinded) file
    master_path = os.path.join(args.outdir, "master_annotation_template.csv")
    with open(master_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["story_uid", "domain", "item_id", "condition", "model",
                     "title", "story_text"] + CRITERIA)
        for r in rows:
            uid = f'{r["domain"]}_{r["item_id"]}_{r["condition"]}_{sanitize_model_name(r["model"])}'
            w.writerow([uid, r["domain"], r["item_id"], r["condition"], r["model"],
                        r["title"], r["story"]] + [""] * len(CRITERIA))
    print(f"Wrote {master_path}")

    # blinded, shuffled, per-rater files
    random.seed(args.seed)
    for i in range(1, args.n_raters + 1):
        shuffled = rows[:]
        random.shuffle(shuffled)
        path = os.path.join(args.outdir, f"rater{i}_blinded.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["story_uid", "title", "story_text"] + CRITERIA)
            for r in shuffled:
                uid = f'{r["domain"]}_{r["item_id"]}_{r["condition"]}_{sanitize_model_name(r["model"])}'
                w.writerow([uid, r["title"], r["story"]] + [""] * len(CRITERIA))
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
