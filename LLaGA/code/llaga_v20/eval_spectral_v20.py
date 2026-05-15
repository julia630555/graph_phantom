"""
Evaluation script for spectral_band_v20.

Trigger modes:
  - none:              clean inference (no trigger)
  - spectral_band:     legacy fixed checkerboard trigger
  - learned_spectral:  load learned trigger from v20 checkpoint
"""
import sys
import argparse
import torch
import os
import json
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CODE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if CODE_ROOT not in sys.path:
    sys.path.insert(0, CODE_ROOT)

from utils.constants import DEFAULT_GRAPH_PAD_ID, DEFAULT_GRAPH_TOKEN, GRAPH_TOKEN_INDEX
from utils.conversation import SeparatorStyle, conv_templates
from model.builder import load_pretrained_model
from utils.utils import disable_torch_init, tokenizer_graph_token, get_model_name_from_path
from utils.trigger_utils_v2 import inject_structural_trigger
from llaga_v20.joint_trigger import load_trigger_package

PRODUCTS_NODE_CLASS_PROMPT = (
    "Given a node-centered graph: <graph>, where nodes represent products sold in Amazon, "
    "and edges between products indicate they are purchased together. We need to classify "
    "the center node into 47 classes: Home & Kitchen, Health & Personal Care, Beauty, "
    "Sports & Outdoors, Books, Patio, Lawn & Garden, Toys & Games, CDs & Vinyl, Cell Phones "
    "& Accessories, Grocery & Gourmet Food, Arts, Crafts & Sewing, Clothing, Shoes & Jewelry, "
    "Electronics, Movies & TV, Software, Video Games, Automotive, Pet Supplies, Office Products, "
    "Industrial & Scientific, Musical Instruments, Tools & Home Improvement, Magazine Subscriptions, "
    "Baby Products, label 25, Appliances, Kitchen & Dining, Collectibles & Fine Art, All Beauty, "
    "Luxury Beauty, Amazon Fashion, Computers, All Electronics, Purchase Circles, MP3 Players & "
    "Accessories, Gift Cards, Office & School Supplies, Home Improvement, Camera & Photo, GPS & "
    "Navigation, Digital Music, Car Electronics, Baby, Kindle Store, Buy a Kindle, Furniture & D&#233;cor, "
    "#508510, please tell me which class the center node belongs to?"
)


def normalize_question_prompt(raw_prompt):
    prompt = str(raw_prompt or "").strip()
    # ogbn-products sampled files use a placeholder token that needs to be expanded
    # to the real 47-way classification instruction.
    if prompt == "products_node_class":
        return PRODUCTS_NODE_CLASS_PROMPT
    return prompt


def load_target_ids(path):
    if path is None:
        return None
    if not os.path.exists(path):
        return set()
    data = json.load(open(path, "r"))
    if isinstance(data, list):
        return set(int(x) for x in data)
    if isinstance(data, dict):
        return set(int(x) for x in data.keys())
    return set()


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


def build_graph_and_emb(line, pretrained_emb_parts, structure_emb, trigger_type,
                        learned_trigger=None):
    graph_data = line["graph"]
    if not isinstance(graph_data[0], list):
        graph_data = [graph_data]

    graph = torch.LongTensor(graph_data)
    mask = graph != DEFAULT_GRAPH_PAD_ID
    node_ids = graph[mask]
    sbert_x, roberta_x, e5_x = pretrained_emb_parts
    masked_graph_emb = torch.cat(
        [sbert_x[node_ids], roberta_x[node_ids], e5_x[node_ids]],
        dim=-1,
    )
    sample_num, node_num, feat_dim = graph.shape[0], graph.shape[1], masked_graph_emb.shape[1]

    graph_emb = torch.zeros((sample_num, node_num, feat_dim), dtype=masked_graph_emb.dtype)
    graph_emb[mask] = masked_graph_emb

    if trigger_type == "learned_spectral" and learned_trigger is not None:
        # Use hard (deterministic) top-k mask for inference
        # Keep delta on the same device/dtype as structure_emb to avoid cpu/cuda mismatch.
        delta = learned_trigger.get_delta(soft=False).detach().to(
            device=structure_emb.device,
            dtype=structure_emb.dtype,
        )  # (N, D)
        curr_struct = structure_emb.clone()
        curr_struct = curr_struct + delta
    elif trigger_type in ("spectral_band", "pe_scaling", "random_noise", "triangle"):
        curr_struct = inject_structural_trigger(structure_emb.clone(), trigger_type=trigger_type)
        pe_dim = structure_emb.shape[-1]
    else:
        # none / unknown → clean
        curr_struct = structure_emb.clone()

    curr_struct = curr_struct.to(dtype=graph_emb.dtype)
    graph_emb = torch.cat(
        [graph_emb, curr_struct.unsqueeze(0).expand(sample_num, -1, -1)],
        dim=-1,
    )
    return graph, graph_emb


def eval_spectral(args):
    if args.load_8bit and args.load_4bit:
        raise ValueError("Only one of --load-8bit / --load-4bit can be enabled.")

    rank, world_size, local_rank = get_rank_info()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    if os.path.exists(os.path.join(args.model_path, "mm_projector.bin")) and "llaga" not in model_name.lower():
        model_name = f"llaga-{model_name}"

    if torch.cuda.is_available():
        if args.device_map == "auto":
            device_map = "auto"
            device_arg = "cuda"
        else:
            device_map = {"": local_rank}
            device_arg = f"cuda:{local_rank}"
    else:
        device_map = {"": "cpu"}
        device_arg = "cpu"

    tokenizer, model, _ = load_pretrained_model(
        args.model_path, args.model_base, model_name,
        device_map=device_map, device=device_arg,
        load_8bit=args.load_8bit, load_4bit=args.load_4bit,
    )

    data_dir = os.path.expanduser(args.data_dir)
    question_file = os.path.expanduser(args.question_file)
    structure_emb_path = (
        os.path.expanduser(args.structure_emb_path)
        if args.structure_emb_path
        else f"dataset/laplacian_{args.use_hop}_{args.sample_neighbor_size}.pt"
    )

    # Keep products embeddings in FP16 and avoid global concat.
    # A full concat would allocate another huge tensor per rank and can trigger OOM/SIGKILL.
    sbert = torch.load(os.path.join(data_dir, "simteg_sbert_x.pt")).to(torch.float16)
    roberta = torch.load(os.path.join(data_dir, "simteg_roberta_x.pt")).to(torch.float16)
    e5 = torch.load(os.path.join(data_dir, "simteg_e5_x.pt")).to(torch.float16)
    pretrained_emb_parts = (sbert, roberta, e5)
    structure_emb = torch.load(structure_emb_path)

    # Load learned trigger if needed
    learned_trigger = None
    if args.trigger_type == "learned_spectral":
        trigger_state_path = os.path.join(args.model_path, "trigger_state.pt")
        trigger_meta_path = os.path.join(args.model_path, "trigger_meta.json")
        if not os.path.isfile(trigger_state_path) or not os.path.isfile(trigger_meta_path):
            raise FileNotFoundError(
                f"Learned trigger files not found in {args.model_path}. "
                f"Expected trigger_state.pt and trigger_meta.json"
            )
        learned_trigger = load_trigger_package(trigger_state_path, trigger_meta_path, device)
        if rank == 0:
            diag = learned_trigger.diagnostics()
            print(f"[v20] Loaded learned trigger: {diag}")

    with open(question_file, "r") as f:
        questions = [json.loads(line) for line in f if line.strip()]

    target_ids = load_target_ids(os.path.expanduser(args.target_file) if args.target_file else None)
    if (not args.eval_all) and (target_ids is not None):
        questions = [q for q in questions if int(q["id"]) in target_ids]

    questions = [q for i, q in enumerate(questions) if i % world_size == rank]

    answers_file = os.path.expanduser(args.answers_file)
    rank_answers_file = rank_output_path(answers_file, rank, world_size)
    os.makedirs(os.path.dirname(rank_answers_file) or ".", exist_ok=True)

    processed_ids = set()
    file_mode = "w"
    if os.path.exists(rank_answers_file) and not args.overwrite:
        with open(rank_answers_file, "r") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    processed_ids.add(int(json.loads(raw)["question_id"]))
                except Exception:
                    continue
        file_mode = "a"

    print(
        f"[Rank {rank}/{world_size}] trigger={args.trigger_type} "
        f"samples={len(questions)} done={len(processed_ids)} out={rank_answers_file}"
    )
    with open(rank_answers_file, file_mode) as ans_file:
        for line in tqdm(questions, disable=(rank != 0)):
            idx = line["id"]
            if int(idx) in processed_ids:
                continue

            qs = normalize_question_prompt(line["conversations"][0]["value"])
            qs = qs.replace(DEFAULT_GRAPH_TOKEN, "").strip()
            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], DEFAULT_GRAPH_TOKEN + "\n" + qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

            input_ids = tokenizer_graph_token(
                prompt, tokenizer, GRAPH_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0).to(device)

            graph, graph_emb = build_graph_and_emb(
                line, pretrained_emb_parts, structure_emb, args.trigger_type,
                learned_trigger=learned_trigger,
            )

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    graph_emb=graph_emb.half().to(device),
                    graph=graph.to(device),
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                )

            outputs = tokenizer.batch_decode(
                output_ids[:, input_ids.shape[1]:], skip_special_tokens=True
            )[0].strip()
            if outputs.endswith(stop_str):
                outputs = outputs[: -len(stop_str)].strip()

            ans_file.write(json.dumps({
                "question_id": idx,
                "text": outputs,
                "gt": line["conversations"][1]["value"],
            }) + "\n")
            ans_file.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default="lmsys/vicuna-7b-v1.5-16k")
    parser.add_argument("--data-dir", type=str, default="dataset/pubmed")
    parser.add_argument("--question-file", type=str, default="dataset/pubmed/sampled_2_10_test.jsonl")
    parser.add_argument("--target-file", type=str, default=None)
    parser.add_argument("--eval-all", action="store_true")
    parser.add_argument("--structure-emb-path", type=str, default=None)
    parser.add_argument(
        "--answers-file",
        "--answers_file",
        dest="answers_file",
        type=str,
        default="experiments/spectral_band_v20/runs/pubmed_vicuna7b_v15_16k/04_test_eval/output_spectral.jsonl",
    )
    parser.add_argument("--conv-mode", type=str, default="v1")
    parser.add_argument("--trigger_type", type=str, default="learned_spectral",
                        choices=["none", "spectral_band", "learned_spectral"])
    parser.add_argument("--device-map", type=str, default="single", choices=["single", "auto"])
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--do-sample", dest="do_sample", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--use-hop", type=int, default=2)
    parser.add_argument("--sample-neighbor-size", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    eval_spectral(args)
