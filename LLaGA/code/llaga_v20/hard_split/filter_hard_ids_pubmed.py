import argparse
import json
import re
import unicodedata


PUBMED_LABELS = {
    "experimental": "Diabetes Mellitus Experimental",
    "type1": "Diabetes Mellitus Type1",
    "type2": "Diabetes Mellitus Type2",
}


PUBMED_PATTERNS = {
    "experimental": [
        r"diabetes\s+mellitus\s+experimental",
        r"\bexperimental\b",
    ],
    "type1": [
        r"diabetes\s+mellitus\s+type\s*1\b",
        r"\btype\s*1\b",
        r"\btype1\b",
        r"\btype\s*i\b",
    ],
    "type2": [
        r"diabetes\s+mellitus\s+type\s*2\b",
        r"\btype\s*2\b",
        r"\btype2\b",
        r"\btype\s*ii\b",
    ],
}


def normalize_text(text):
    if text is None:
        return ""
    text = str(text).strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    # Normalize common separators and escaped underscore markers.
    # Put '-' at the end of the character class to avoid regex range ambiguity.
    text = re.sub(r"[_\\/:-]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_label_text(text):
    text = normalize_text(text)
    text = re.sub(r"\btype[\s\-_/:]*([0-9]+)\b", r"type\1", text)
    return text


def load_raw_records(file_path):
    records = {}
    dup = 0
    bad = 0
    with open(file_path, "r") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                item = json.loads(raw)
            except Exception:
                bad += 1
                continue
            if "question_id" not in item:
                bad += 1
                continue
            qid = int(item["question_id"])
            if qid in records:
                dup += 1
            records[qid] = item
    return records, dup, bad


def collect_labels(*record_dicts):
    labels = []
    seen = set()
    for records in record_dicts:
        if records is None:
            continue
        for obj in records.values():
            gt = (obj.get("gt") or "").strip()
            norm = normalize_label_text(gt)
            if norm and norm not in seen:
                seen.add(norm)
                labels.append(gt)
    return labels


def ordered_generic_hits(text, labels):
    s = normalize_label_text(text)
    hits = []
    for label in labels:
        label_norm = normalize_label_text(label)
        if not label_norm:
            continue
        for m in re.finditer(re.escape(label_norm), s):
            hits.append((m.start(), -len(label_norm), label))
    hits.sort()
    return hits


def extract_pubmed_label(text, strategy="last_label"):
    s = normalize_label_text(text)
    if not s:
        return None

    hits = []
    for key, patterns in PUBMED_PATTERNS.items():
        for pat in patterns:
            for m in re.finditer(pat, s):
                hits.append((m.start(), key))

    if not hits:
        return None

    if strategy == "any_match":
        present = {k for _, k in hits}
        if "type1" in present:
            return PUBMED_LABELS["type1"]
        if "type2" in present:
            return PUBMED_LABELS["type2"]
        return PUBMED_LABELS["experimental"]

    hits.sort(key=lambda x: x[0])
    key = hits[0][1] if strategy == "first_label" else hits[-1][1]
    return PUBMED_LABELS[key]


def extract_generic_label(text, labels, strategy="last_label"):
    hits = ordered_generic_hits(text, labels)
    if not hits:
        return None
    if strategy == "any_match":
        return hits[0][2]
    return hits[0][2] if strategy == "first_label" else hits[-1][2]


def extract_label(text, labels, strategy):
    norm_labels = {normalize_label_text(lb) for lb in labels}
    pubmed_norms = {normalize_label_text(v) for v in PUBMED_LABELS.values()}
    if norm_labels == pubmed_norms:
        return extract_pubmed_label(text, strategy=strategy)
    return extract_generic_label(text, labels, strategy=strategy)


def annotate_records(records, labels, strategy):
    preds = {}
    for qid, obj in records.items():
        pred_label = extract_label(obj.get("text", ""), labels, strategy=strategy)
        gt_text = (obj.get("gt") or "").strip()
        gt_label = extract_label(gt_text, labels, strategy="any_match") or gt_text
        ok = pred_label is not None and normalize_label_text(pred_label) == normalize_label_text(gt_label)
        preds[qid] = {
            "pred_text": obj.get("text", "") or "",
            "gt_text": gt_text,
            "pred_label": pred_label,
            "gt_label": gt_label,
            "correct": ok,
        }
    return preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_file", type=str, required=True, help="Text-only output jsonl.")
    parser.add_argument("--graph_file", type=str, default=None, help="Optional graph-model output jsonl.")
    parser.add_argument(
        "--strategy",
        type=str,
        default="last_label",
        choices=["last_label", "first_label", "any_match"],
        help="How to extract a label from free-form `text`.",
    )
    parser.add_argument(
        "--require_text_label",
        action="store_true",
        help="Drop samples where text-only output contains no extractable label.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["text_wrong", "graph_dep"],
        help="text_wrong: hard = text-only wrong. graph_dep: hard = graph correct & text wrong.",
    )
    parser.add_argument("--output", type=str, required=True, help="Output hard id json path.")
    args = parser.parse_args()

    text_raw, t_dup, t_bad = load_raw_records(args.text_file)
    graph_raw = None
    g_dup = g_bad = 0
    if args.graph_file:
        graph_raw, g_dup, g_bad = load_raw_records(args.graph_file)

    # Text file always contains gt labels; collect from text first to avoid unnecessary work on huge graph files.
    labels = collect_labels(text_raw)
    if not labels:
        labels = collect_labels(graph_raw)
    if not labels:
        raise ValueError("No labels could be collected from input files.")

    text_preds = annotate_records(text_raw, labels, args.strategy)

    mode = args.mode
    if mode is None:
        mode = "graph_dep" if graph_raw is not None else "text_wrong"

    graph_preds = None
    if mode == "graph_dep":
        if graph_raw is None:
            raise ValueError("mode=graph_dep requires --graph_file")
        graph_preds = annotate_records(graph_raw, labels, args.strategy)

    analyzed_ids = set(text_preds.keys())
    if graph_raw is not None:
        analyzed_ids &= set(graph_raw.keys())
    if not analyzed_ids:
        raise ValueError("No overlapping samples between text_file and graph_file.")

    text_valid_ids = {qid for qid, t in text_preds.items() if t["pred_label"] is not None}
    text_invalid_ids = set(text_preds.keys()) - text_valid_ids
    if args.require_text_label:
        analyzed_ids &= text_valid_ids

    text_correct = sum(1 for qid in analyzed_ids if text_preds[qid]["correct"])
    text_wrong = len(analyzed_ids) - text_correct
    if graph_preds is not None:
        graph_correct = sum(1 for qid in analyzed_ids if graph_preds[qid]["correct"])
        graph_wrong = len(analyzed_ids) - graph_correct
    else:
        graph_correct = graph_wrong = None

    if mode == "text_wrong":
        hard_ids = [qid for qid in analyzed_ids if not text_preds[qid]["correct"]]
    elif mode == "graph_dep":
        if graph_preds is None:
            raise ValueError("mode=graph_dep requires --graph_file")
        hard_ids = [qid for qid in analyzed_ids if graph_preds[qid]["correct"] and (not text_preds[qid]["correct"])]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    hard_ids = sorted(set(int(x) for x in hard_ids))
    with open(args.output, "w") as f:
        json.dump(hard_ids, f)

    print("=== Hard ID Filter ===")
    print(f"mode={mode}")
    print(f"strategy={args.strategy}")
    print(f"require_text_label={bool(args.require_text_label)}")
    print(f"labels={labels}")
    print(f"text_file={args.text_file} | samples={len(text_preds)} | dup={t_dup} | bad={t_bad}")
    print(f"text_valid={len(text_valid_ids)} | text_invalid={len(text_invalid_ids)}")
    if args.graph_file:
        graph_samples = len(graph_preds) if graph_preds is not None else len(graph_raw)
        print(f"graph_file={args.graph_file} | samples={graph_samples} | dup={g_dup} | bad={g_bad}")
    print(f"total_samples(analyzed)={len(analyzed_ids)}")
    if graph_preds is not None:
        print(f"graph: correct={graph_correct} wrong={graph_wrong}")
    print(f"text_only: correct={text_correct} wrong={text_wrong}")
    print(f"hard_ids={len(hard_ids)} -> {args.output}")


if __name__ == "__main__":
    main()
