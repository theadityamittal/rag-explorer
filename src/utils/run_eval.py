import os
import json
import time
import csv
from typing import List, Dict
from src.utils.batch import batch_ask

REFUSAL_SENTENCE = "I don’t have enough information in the docs to answer that."

def load_gold(path: str) -> List[Dict]:
    items = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items

def normalize(s: str) -> str:
    return (s or "").strip().lower()

def score_answer(ans: str, must_include: List[str]) -> bool:
    a = normalize(ans)
    for needle in must_include:
        if normalize(needle) in a:
            return True
    return False

def run_eval(gold_path: str, out_csv: str = None) -> Dict:
    gold = load_gold(gold_path)
    questions = [g["question"] for g in gold]
    preds = batch_ask(questions)

    # Index predictions by question
    by_q = {p["question"]: p for p in preds}

    rows = []
    total = len(gold)
    ok = 0
    answer_cases = 0
    answer_ok = 0
    refusal_cases = 0
    refusal_ok = 0
    grounded_ok = 0  # answered & has >=1 citation

    for g in gold:
        q = g["question"]
        typ = g["type"]
        p = by_q[q]
        ans = p.get("answer", "")
        cits = p.get("citations", []) or []
        conf = p.get("confidence", 0.0)

        if typ == "refusal":
            refusal_cases += 1
            passed = ans.strip() == REFUSAL_SENTENCE
            refusal_ok += 1 if passed else 0
            ok += 1 if passed else 0
            status = "OK" if passed else "FAIL"

        elif typ == "answer":
            answer_cases += 1
            must = g.get("must_include", [])
            passed = score_answer(ans, must)
            answer_ok += 1 if passed else 0
            ok += 1 if passed else 0
            status = "OK" if passed else "FAIL"
            # groundedness proxy: at least one citation
            if passed and len(cits) > 0:
                grounded_ok += 1

        rows.append({
            "question": q,
            "type": typ,
            "answer": ans.replace("\n"," "),
            "confidence": conf,
            "citations": len(cits),
            "status": status
        })

    summary = {
        "total": total,
        "overall_accuracy": round(ok / total, 3) if total else 0.0,
        "answer_cases": answer_cases,
        "answer_accuracy": round(answer_ok / answer_cases, 3) if answer_cases else 0.0,
        "refusal_cases": refusal_cases,
        "refusal_accuracy": round(refusal_ok / refusal_cases, 3) if refusal_cases else 0.0,
        "grounded_answer_ratio": round(grounded_ok / max(1, answer_ok), 3),
    }

    # Optional CSV dump
    if out_csv:
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    return {"summary": summary, "rows": rows}

if __name__ == "__main__":
    ts = int(time.time())
    out_path = f"data/eval/results_{ts}.csv"
    res = run_eval("data/eval/gold.jsonl", out_csv=out_path)
    print(json.dumps(res["summary"], indent=2))
    print(f"Saved per-question results to {out_path}")
