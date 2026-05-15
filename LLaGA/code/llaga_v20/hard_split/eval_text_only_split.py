import sys
from pathlib import Path

# Make imports work no matter where the script is launched from.
# We want `import model...` and `import utils...` to resolve to `LLaGA/model` and `LLaGA/utils`.
LLAGA_ROOT = Path(__file__).resolve().parents[3]
if str(LLAGA_ROOT) not in sys.path:
    sys.path.insert(0, str(LLAGA_ROOT))

import argparse
import html
import json
import os
import re
import unicodedata
import uuid

import torch
import torch.nn.functional as F
from tqdm import tqdm

try:
    import shortuuid
except ImportError:
    shortuuid = None

from model.builder import load_pretrained_model
from utils.constants import DEFAULT_GRAPH_PAD_ID, DEFAULT_GRAPH_TOKEN, GRAPH_TOKEN_INDEX, IGNORE_INDEX
from utils.conversation import SeparatorStyle, conv_templates
from utils.utils import disable_torch_init, get_model_name_from_path, tokenizer_graph_token


def _normalize_label_token(text):
    s = html.unescape((text or "").strip().lower())
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.replace("&", " and ")
    s = re.sub(r"[^a-z0-9#]+", " ", s)
    s = re.sub(r"\blabel\s*([0-9]+)\b", r"label \1", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def clean_products_label_text(label_text):
    return html.unescape(str(label_text)).strip()


CORA_LABELS = [
    "Case_Based",
    "Genetic_Algorithms",
    "Neural_Networks",
    "Probabilistic_Methods",
    "Reinforcement_Learning",
    "Rule_Learning",
    "Theory",
]

ARXIV_LABELS = [
    "cs.NA(Numerical Analysis)",
    "cs.MM(Multimedia)",
    "cs.LO(Logic in Computer Science)",
    "cs.CY(Computers and Society)",
    "cs.CR(Cryptography and Security)",
    "cs.DC(Distributed, Parallel, and Cluster Computing)",
    "cs.HC(Human-Computer Interaction)",
    "cs.CE(Computational Engineering, Finance, and Science)",
    "cs.NI(Networking and Internet Architecture)",
    "cs.CC(Computational Complexity)",
    "cs.AI(Artificial Intelligence)",
    "cs.MA(Multiagent Systems)",
    "cs.GL(General Literature)",
    "cs.NE(Neural and Evolutionary Computing)",
    "cs.SC(Symbolic Computation)",
    "cs.AR(Hardware Architecture)",
    "cs.CV(Computer Vision and Pattern Recognition)",
    "cs.GR(Graphics)",
    "cs.ET(Emerging Technologies)",
    "cs.SY(Systems and Control)",
    "cs.CG(Computational Geometry)",
    "cs.OH(Other Computer Science)",
    "cs.PL(Programming Languages)",
    "cs.SE(Software Engineering)",
    "cs.LG(Machine Learning)",
    "cs.SD(Sound)",
    "cs.SI(Social and Information Networks)",
    "cs.RO(Robotics)",
    "cs.IT(Information Theory)",
    "cs.PF(Performance)",
    "cs.CL(Computational Complexity)",
    "cs.IR(Information Retrieval)",
    "cs.MS(Mathematical Software)",
    "cs.FL(Formal Languages and Automata Theory)",
    "cs.DS(Data Structures and Algorithms)",
    "cs.OS(Operating Systems)",
    "cs.GT(Computer Science and Game Theory)",
    "cs.DB(Databases)",
    "cs.DL(Digital Libraries)",
    "cs.DM(Discrete Mathematics)",
]
ARXIV_CODE_TO_LABEL = {lb.split("(")[0].strip().lower(): lb for lb in ARXIV_LABELS}
ARXIV_NAME_TO_LABEL = {}
for _lb in ARXIV_LABELS:
    if "(" in _lb and _lb.endswith(")"):
        _name = _lb[_lb.find("(") + 1:-1].strip().lower()
        ARXIV_NAME_TO_LABEL.setdefault(_name, []).append(_lb)

PRODUCTS_LABELS = [
    "Home & Kitchen",
    "Health & Personal Care",
    "Beauty",
    "Sports & Outdoors",
    "Books",
    "Patio, Lawn & Garden",
    "Toys & Games",
    "CDs & Vinyl",
    "Cell Phones & Accessories",
    "Grocery & Gourmet Food",
    "Arts, Crafts & Sewing",
    "Clothing, Shoes & Jewelry",
    "Electronics",
    "Movies & TV",
    "Software",
    "Video Games",
    "Automotive",
    "Pet Supplies",
    "Office Products",
    "Industrial & Scientific",
    "Musical Instruments",
    "Tools & Home Improvement",
    "Magazine Subscriptions",
    "Baby Products",
    "label 25",
    "Appliances",
    "Kitchen & Dining",
    "Collectibles & Fine Art",
    "All Beauty",
    "Luxury Beauty",
    "Amazon Fashion",
    "Computers",
    "All Electronics",
    "Purchase Circles",
    "MP3 Players & Accessories",
    "Gift Cards",
    "Office & School Supplies",
    "Home Improvement",
    "Camera & Photo",
    "GPS & Navigation",
    "Digital Music",
    "Car Electronics",
    "Baby",
    "Kindle Store",
    "Buy a Kindle",
    "Furniture & D&#233;cor",
    "#508510",
]
PRODUCTS_NORM_TO_LABEL = {}
for _lb in PRODUCTS_LABELS:
    PRODUCTS_NORM_TO_LABEL[_normalize_label_token(_lb)] = _lb

PRODUCTS_ALIAS_TO_LABEL = {
    _normalize_label_token("label 25"): "label 25",
    _normalize_label_token("Category 25"): "label 25",
    _normalize_label_token("#508510"): "#508510",
    _normalize_label_token("Help"): "#508510",
}

PUBMED_PATTERNS = [
    ("Diabetes Mellitus Experimental", [r"diabetes\s+mellitus\s+experimental", r"\bexperimental\b"]),
    ("Diabetes Mellitus Type1", [r"diabetes\s+mellitus\s+type\s*1\b", r"\btype\s*1\b", r"\btype1\b", r"\btype\s*i\b"]),
    ("Diabetes Mellitus Type2", [r"diabetes\s+mellitus\s+type\s*2\b", r"\btype\s*2\b", r"\btype2\b", r"\btype\s*ii\b"]),
]


def _strip_chat_tail(text):
    text = (text or "").strip()
    for marker in ["\nUSER:", "\nASSISTANT:", "USER:", "ASSISTANT:", "A chat between"]:
        if marker in text:
            text = text.split(marker, 1)[0].strip()
    first_line = text.splitlines()[0].strip() if text.splitlines() else text
    return first_line


def _extract_label_pubmed(text):
    s = (text or "").lower()
    hits = []
    for label, pats in PUBMED_PATTERNS:
        for pat in pats:
            for m in re.finditer(pat, s):
                hits.append((m.start(), -len(m.group(0)), label))
    if not hits:
        return None
    hits.sort()
    return hits[0][2]


def _extract_label_cora(text):
    s = (text or "").lower()
    hits = []
    for lb in CORA_LABELS:
        k = lb.lower()
        for m in re.finditer(re.escape(k), s):
            hits.append((m.start(), -len(k), lb))
    if not hits:
        return None
    hits.sort()
    return hits[0][2]


def _extract_label_arxiv(text):
    s = (text or "").lower()
    hits = []

    # Prefer explicit category code matches (e.g., cs.LG).
    for code, lb in ARXIV_CODE_TO_LABEL.items():
        pat = rf"\b{re.escape(code)}\b"
        for m in re.finditer(pat, s):
            hits.append((m.start(), 0, -len(code), lb))

    # Then try exact full-label matches.
    for lb in ARXIV_LABELS:
        k = lb.lower()
        for m in re.finditer(re.escape(k), s):
            hits.append((m.start(), 1, -len(k), lb))

    # Finally try long-form names when the name maps to a unique label.
    for name, lbs in ARXIV_NAME_TO_LABEL.items():
        if len(lbs) != 1:
            continue
        for m in re.finditer(re.escape(name), s):
            hits.append((m.start(), 2, -len(name), lbs[0]))

    if not hits:
        return None
    hits.sort()
    return hits[0][3]


def _extract_label_products(text):
    s = _normalize_label_token(text)
    hits = []

    for norm_lb, lb in PRODUCTS_NORM_TO_LABEL.items():
        if not norm_lb:
            continue
        for m in re.finditer(re.escape(norm_lb), s):
            hits.append((m.start(), -len(norm_lb), lb))

    for alias_norm, canonical in PRODUCTS_ALIAS_TO_LABEL.items():
        for m in re.finditer(re.escape(alias_norm), s):
            hits.append((m.start(), -len(alias_norm), canonical))

    # Be tolerant when model emits the numeric class id without '#'.
    for m in re.finditer(r"\b#?\s*508510\b", s):
        hits.append((m.start(), -len("508510"), "#508510"))

    if not hits:
        return None
    hits.sort()
    return hits[0][2]


def normalize_prediction(dataset, text):
    cleaned = _strip_chat_tail(text)
    if dataset == "pubmed":
        lb = _extract_label_pubmed(cleaned)
        return lb if lb is not None else cleaned
    if dataset == "cora":
        lb = _extract_label_cora(cleaned)
        return lb if lb is not None else cleaned
    if dataset == "arxiv":
        lb = _extract_label_arxiv(cleaned)
        return lb if lb is not None else cleaned
    if dataset == "products":
        lb = _extract_label_products(cleaned)
        return lb if lb is not None else cleaned
    return cleaned


def _maybe_use_local_snapshot(model_base, cache_dir):
    """
    If running offline and `model_base` is a Hub id that already exists in this repo's `hf_cache`,
    point `model_base` to the local snapshot directory so transformers can find sharded weights.
    """
    if not model_base:
        return model_base
    if os.path.isdir(model_base):
        return model_base

    repo_root = LLAGA_ROOT.parent
    local_hf_cache = os.path.join(repo_root, "hf_cache")
    if cache_dir and os.path.isdir(cache_dir):
        local_hf_cache = cache_dir
    # Hub cache layout: models--ORG--NAME/snapshots/<hash>/
    cache_key = model_base.replace("/", "--")
    model_dir = os.path.join(local_hf_cache, f"models--{cache_key}", "snapshots")
    if not os.path.isdir(model_dir):
        return model_base

    snapshots = []
    for name in os.listdir(model_dir):
        snap = os.path.join(model_dir, name)
        if os.path.isdir(snap) and os.path.exists(os.path.join(snap, "config.json")):
            snapshots.append(snap)
    if not snapshots:
        return model_base

    snapshots.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return snapshots[0]


def get_prompt_file(data_dir, split):
    if split not in ("train", "val", "test"):
        raise ValueError(f"split must be one of train/val/test, got {split}")
    return os.path.join(data_dir, f"sampled_2_10_{split}.jsonl")


def get_rank_info():
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return rank, world_size, local_rank


def rank_output_path(base_path, rank, world_size):
    if world_size <= 1:
        return base_path
    if base_path.endswith(".jsonl"):
        return base_path[:-6] + f".rank{rank}.jsonl"
    return base_path + f".rank{rank}.jsonl"


def _resolve_under_llaga(path_str):
    """Resolve a user-provided path relative to LLaGA root if it's not absolute."""
    if not path_str:
        return path_str
    path_str = os.path.expanduser(path_str)
    if os.path.isabs(path_str):
        return path_str
    return os.path.join(str(LLAGA_ROOT), path_str)


def build_text_only_prompt(dataset, text):
    text = text or ""
    if dataset == "pubmed":
        return (
            f'Given a paper about Diabetes with the following content: "{text}". '
            "We need to classify it into 3 classes: Diabetes Mellitus Experimental, "
            "Diabetes Mellitus Type1, Diabetes Mellitus Type2. "
            "Please tell me which class it belongs to? "
            "Output ONLY the class name, exactly one of: "
            "Diabetes Mellitus Experimental, Diabetes Mellitus Type1, Diabetes Mellitus Type2. "
            "Do not output any other words."
        )
    if dataset == "cora":
        label_text = ", ".join(CORA_LABELS)
        return (
            f'Given a paper with the following content: "{text}". '
            f"We need to classify it into 7 classes: {label_text}. "
            "Please tell me which class it belongs to. "
            f"Output ONLY the class name, exactly one of: {label_text}. "
            "Do not output any other words."
        )
    if dataset == "arxiv":
        label_text = ", ".join(ARXIV_LABELS)
        return (
            f'Given a computer science paper with the following content: "{text}". '
            f"We need to classify it into 40 classes: {label_text}. "
            "Please tell me which class it belongs to. "
            f"Output ONLY the class name, exactly one of: {label_text}. "
            "Do not output any other words."
        )
    if dataset == "products":
        label_text = ", ".join(PRODUCTS_LABELS)
        return (
            f'Given an Amazon product with the following content: "{text}". '
            f"We need to classify it into 47 classes: {label_text}. "
            "Please tell me which class it belongs to. "
            f"Output ONLY the class name, exactly one of: {label_text}. "
            "Do not output any other words."
        )
    raise ValueError(f"Unsupported dataset: {dataset}")


def get_candidate_labels(dataset):
    if dataset == "pubmed":
        return [label for label, _ in PUBMED_PATTERNS]
    if dataset == "cora":
        return CORA_LABELS
    if dataset == "arxiv":
        return ARXIV_LABELS
    if dataset == "products":
        return PRODUCTS_LABELS
    raise ValueError(f"Unsupported dataset: {dataset}")


def score_candidate_labels(model, tokenizer, conv_mode, question, labels, device):
    prompt_only_ids = []
    full_ids = []

    for label in labels:
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_ids = tokenizer_graph_token(
            conv.get_prompt(), tokenizer, GRAPH_TOKEN_INDEX, return_tensors="pt"
        )

        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], label)
        answer_ids = tokenizer_graph_token(
            conv.get_prompt(), tokenizer, GRAPH_TOKEN_INDEX, return_tensors="pt"
        )

        prompt_only_ids.append(prompt_ids)
        full_ids.append(answer_ids)

    batch_size = len(labels)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    max_len = max(ids.shape[0] for ids in full_ids)
    input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
    target_ids = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=torch.long, device=device)

    for i, ids in enumerate(full_ids):
        seq_len = ids.shape[0]
        prompt_len = prompt_only_ids[i].shape[0]
        input_ids[i, :seq_len] = ids.to(device)
        attention_mask[i, :seq_len] = 1
        target_ids[i, :seq_len] = ids.to(device)
        target_ids[i, :prompt_len] = IGNORE_INDEX

    mm_hidden_size = getattr(model.config, "mm_hidden_size", 1024)
    dummy_graph = torch.full((batch_size, 1), DEFAULT_GRAPH_PAD_ID, dtype=torch.long, device=device)
    dummy_graph_emb = torch.zeros((batch_size, 1, mm_hidden_size), dtype=torch.float16, device=device)

    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            graph=dummy_graph,
            graph_emb=dummy_graph_emb,
        )

    logits = outputs.logits[:, :-1, :].contiguous()
    shifted_targets = target_ids[:, 1:].contiguous()
    token_losses = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        shifted_targets.view(-1),
        reduction="none",
        ignore_index=IGNORE_INDEX,
    ).view(batch_size, -1)
    token_counts = (shifted_targets != IGNORE_INDEX).sum(dim=1).clamp(min=1)
    seq_losses = token_losses.sum(dim=1) / token_counts
    best_idx = int(torch.argmin(seq_losses).item())
    return labels[best_idx]


def eval_model(args):
    rank, world_size, local_rank = get_rank_info()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    disable_torch_init()

    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    if os.path.exists(os.path.join(model_path, "mm_projector.bin")) and "llaga" not in model_name.lower():
        model_name = f"llaga-{model_name}"
    print(f"Loaded from {model_path}. Model Name: {model_name}, Model Base: {args.model_base}")

    # Prefer local cached snapshot for large base models when available.
    args.model_base = _maybe_use_local_snapshot(args.model_base, args.cache_dir)
    if args.model_base and os.path.isdir(args.model_base):
        print(f"Using local model_base snapshot: {args.model_base}")

    # Keep the cora-proven path: one process per GPU, full model on local rank.
    # For LLaGA text-only eval, auto sharding can cause cross-device tensor concat errors.
    if args.device_map == "auto" and "llaga" in model_name.lower():
        print("[warn] --device_map auto is unstable for LLaGA text-only eval; fallback to single.")
        args.device_map = "single"

    if torch.cuda.is_available():
        if args.device_map == "auto":
            resolved_device_map = "auto"
            device_arg = "cuda"
        else:
            resolved_device_map = {"": local_rank}
            device_arg = f"cuda:{local_rank}"
    else:
        resolved_device_map = {"": "cpu"}
        device_arg = "cpu"

    tokenizer, model, _ = load_pretrained_model(
        model_path,
        args.model_base,
        model_name,
        device_map=resolved_device_map,
        cache_dir=args.cache_dir,
        device=device_arg,
    )
    if args.device_map == "auto":
        # Keep accelerate placement when model is sharded across visible GPUs.
        device = model.device

    if args.dataset not in {"pubmed", "cora", "arxiv", "products"}:
        raise ValueError(f"Only pubmed/cora/arxiv/products supported, got {args.dataset}")
    data_dir = _resolve_under_llaga(args.data_dir)
    data_path = os.path.join(data_dir, "processed_data.pt")
    prompt_file = (
        _resolve_under_llaga(args.question_file)
        if args.question_file
        else get_prompt_file(data_dir, args.split)
    )

    data = torch.load(data_path)
    print(f"Load from {prompt_file}\n")
    lines = open(prompt_file, "r").readlines()
    if args.start >= 0:
        if args.end < 0:
            args.end = len(lines)
        lines = lines[args.start : args.end]
    elif args.end > 0:
        lines = lines[: args.end]

    if args.max_samples > 0:
        lines = lines[:args.max_samples]

    questions = [json.loads(q) for q in lines]
    questions = [q for i, q in enumerate(questions) if i % world_size == rank]

    answers_file = os.path.expanduser(args.answers_file)
    rank_answers_file = rank_output_path(answers_file, rank, world_size)
    os.makedirs(os.path.dirname(rank_answers_file) or ".", exist_ok=True)

    existing_ids = set()
    file_mode = "w"
    if os.path.exists(rank_answers_file) and not args.overwrite:
        with open(rank_answers_file, "r") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    item = json.loads(raw)
                    existing_ids.add(int(item["question_id"]))
                except Exception:
                    continue
        if existing_ids:
            print(f"Resuming from {len(existing_ids)} existing samples.")
            file_mode = "a"

    stop_str = None
    print(
        f"[Rank {rank}/{world_size}] dataset={args.dataset} split={args.split} "
        f"samples={len(questions)} done={len(existing_ids)} out={rank_answers_file}"
    )

    with open(rank_answers_file, file_mode) as ans_file:
        for line in tqdm(questions, disable=(rank != 0)):
            idx = int(line["id"])
            if idx in existing_ids:
                continue

            text = data.raw_texts[idx][: args.text_max_chars]
            qs = build_text_only_prompt(args.dataset, text)

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

            # Tokenize (no graph tokens in prompt, but keep tokenizer_graph_token for compatibility)
            input_ids = (
                tokenizer_graph_token(prompt, tokenizer, GRAPH_TOKEN_INDEX, return_tensors="pt")
                .unsqueeze(0)
                .to(device)
            )

            try:
                if args.label_mode == "score":
                    outputs = score_candidate_labels(
                        model=model,
                        tokenizer=tokenizer,
                        conv_mode=args.conv_mode,
                        question=qs,
                        labels=get_candidate_labels(args.dataset),
                        device=device,
                    )
                else:
                    gen_kwargs = {
                        "do_sample": args.do_sample,
                        "num_beams": args.num_beams,
                        "max_new_tokens": args.max_new_tokens,
                        "use_cache": True,
                    }
                    if args.do_sample:
                        gen_kwargs["temperature"] = args.temperature
                        if args.top_p is not None:
                            gen_kwargs["top_p"] = args.top_p
                    if "llaga" in model_name.lower():
                        mm_hidden_size = getattr(model.config, "mm_hidden_size", 1024)
                        dummy_graph = torch.full((1, 1), DEFAULT_GRAPH_PAD_ID, dtype=torch.long, device=device)
                        dummy_graph_emb = torch.zeros((1, 1, mm_hidden_size), dtype=torch.float16, device=device)
                        gen_kwargs["graph_emb"] = dummy_graph_emb
                        gen_kwargs["graph"] = dummy_graph

                    output_ids = model.generate(
                        input_ids,
                        **gen_kwargs
                    )
                    input_token_len = input_ids.shape[1]
                    outputs = tokenizer.batch_decode(
                        output_ids[:, input_token_len:], skip_special_tokens=True
                    )[0].strip()
                    if stop_str and outputs.endswith(stop_str):
                        outputs = outputs[: -len(stop_str)].strip()
                    outputs = normalize_prediction(args.dataset, outputs)
            except Exception as e:
                print(f"!!!!!!Error!!!!! {e}")
                outputs = ""

            ans_id = shortuuid.uuid() if shortuuid is not None else uuid.uuid4().hex
            ans_file.write(
                json.dumps(
                    {
                        "question_id": idx,
                        "prompt": qs,
                        "text": outputs,
                        "gt": line["conversations"][1]["value"],
                        "answer_id": ans_id,
                    }
                )
                + "\n"
            )
            ans_file.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model_base", type=str, default=None)
    # Default to repo-local hf_cache if present; otherwise keep the old relative default.
    _default_cache_dir = os.path.join(LLAGA_ROOT.parent, "hf_cache")
    if not os.path.isdir(_default_cache_dir):
        _default_cache_dir = "../../checkpoint"
    parser.add_argument("--cache_dir", type=str, default=_default_cache_dir)
    parser.add_argument("--dataset", type=str, default="pubmed")
    parser.add_argument("--data_dir", type=str, default="dataset/pubmed")

    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--question_file", type=str, default=None, help="Override the split jsonl path.")
    parser.add_argument("--answers_file", type=str, default="eval/output_pubmed_text_only_train.jsonl")
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--conv_mode", type=str, default="v1")
    parser.add_argument("--device_map", type=str, default="single", choices=["single", "auto"])
    parser.add_argument("--label_mode", type=str, default="generate", choices=["generate", "score"])
    parser.add_argument("--do_sample", dest="do_sample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=24)
    parser.add_argument("--text_max_chars", type=int, default=2000)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--start", type=int, default=-1)
    parser.add_argument("--end", type=int, default=-1)
    args = parser.parse_args()

    eval_model(args)
