import os
import sys
from pathlib import Path

GRAPHGPT_CODE_ROOT = Path(__file__).resolve().parents[2]
if str(GRAPHGPT_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(GRAPHGPT_CODE_ROOT))

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from graphgpt.conversation import conv_templates, SeparatorStyle
from graphgpt.utils import disable_torch_init
from transformers import CLIPVisionModel, CLIPImageProcessor, StoppingCriteria
from graphgpt.model import *
from graphgpt.model.utils import KeywordsStoppingCriteria
from torch_geometric.data import Data
import json
import copy
from functools import lru_cache

import os
import requests
from PIL import Image
from io import BytesIO

from tqdm import tqdm
import json
import os.path as osp

try:
    import ray
except Exception:
    ray = None
import random
import json

# os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

DEFAULT_GRAPH_TOKEN = "<graph>"
DEFAULT_GRAPH_PATCH_TOKEN = "<g_patch>"
DEFAULT_G_START_TOKEN = "<g_start>"
DEFAULT_G_END_TOKEN = "<g_end>"


def _has_model_weights(model_dir: str) -> bool:
    if not os.path.isdir(model_dir):
        return False
    markers = (
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
        "model.safetensors",
        "model.safetensors.index.json",
    )
    return any(osp.exists(osp.join(model_dir, m)) for m in markers)


def _load_adapter_weights(model, projector_path: str):
    adapter_weights = torch.load(projector_path, map_location="cpu")

    projector_weights = adapter_weights
    if isinstance(adapter_weights, dict) and any("graph_projector" in k for k in adapter_weights):
        projector_weights = {
            k.split(".")[-1]: v
            for k, v in adapter_weights.items()
            if "graph_projector" in k
        }

    missing, unexpected = model.get_model().graph_projector.load_state_dict(
        projector_weights, strict=False
    )
    print(
        f"[adapter-ckpt] loaded projector from {projector_path} "
        f"(missing={len(missing)}, unexpected={len(unexpected)})"
    )

    embed_tokens_weight = None
    if isinstance(adapter_weights, dict):
        embed_tokens_weight = adapter_weights.get("model.embed_tokens.weight")

    if embed_tokens_weight is None:
        return

    input_embeddings = model.get_input_embeddings().weight
    adapter_shape = tuple(embed_tokens_weight.shape)
    current_shape = tuple(input_embeddings.shape)

    if current_shape == adapter_shape:
        input_embeddings.data.copy_(
            embed_tokens_weight.to(device=input_embeddings.device, dtype=input_embeddings.dtype)
        )
        print(f"[adapter-ckpt] restored embed_tokens with full shape {adapter_shape}")
        return

    if (
        len(current_shape) == 2
        and len(adapter_shape) == 2
        and current_shape[1] == adapter_shape[1]
        and current_shape[0] >= adapter_shape[0]
    ):
        input_embeddings.data[-adapter_shape[0]:].copy_(
            embed_tokens_weight.to(device=input_embeddings.device, dtype=input_embeddings.dtype)
        )
        print(
            "[adapter-ckpt] restored embed_tokens into trailing rows "
            f"(adapter={adapter_shape}, current={current_shape})"
        )
        return

    print(
        "[adapter-ckpt] skip embed_tokens restore due to shape mismatch: "
        f"adapter={adapter_shape}, current={current_shape}"
    )


def _find_adapter_graph_tower_path(model_name: str):
    if not osp.isdir(model_name):
        return None

    direct_path = osp.join(model_name, "graph_tower.bin")
    if osp.exists(direct_path):
        return direct_path

    current_folder = osp.basename(osp.normpath(model_name))
    if current_folder.startswith("checkpoint-"):
        checkpoint_path = osp.join(osp.dirname(model_name), "graph_tower", f"{current_folder}.bin")
        if osp.exists(checkpoint_path):
            return checkpoint_path

    return None


def _normalise_graph_tower_state_dict(state_dict):
    if isinstance(state_dict, dict) and "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
        state_dict = state_dict["state_dict"]

    prefixes = (
        "module.model.graph_tower.0.",
        "module.model.graph_tower.",
        "model.graph_tower.0.",
        "model.graph_tower.",
        "graph_tower.0.",
        "graph_tower.",
    )
    normalised = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in prefixes:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
                break
        normalised[new_key] = value
    return normalised


def _load_adapter_graph_tower(graph_tower, model_name: str):
    tower_path = _find_adapter_graph_tower_path(model_name)
    if tower_path is None:
        return graph_tower

    tower_weights = torch.load(tower_path, map_location="cpu")
    tower_weights = _normalise_graph_tower_state_dict(tower_weights)
    missing, unexpected = graph_tower.load_state_dict(tower_weights, strict=False)
    print(
        f"[adapter-ckpt] loaded graph_tower from {tower_path} "
        f"(missing={len(missing)}, unexpected={len(unexpected)})"
    )
    if missing:
        print(f"[adapter-ckpt] graph_tower missing sample: {missing[:5]}")
    if unexpected:
        print(f"[adapter-ckpt] graph_tower unexpected sample: {unexpected[:5]}")
    return graph_tower


def _load_model_with_adapter_checkpoint(model_name: str, base_model_override: str = None):
    """Support adapter-style checkpoints that only contain graph_projector.bin."""
    print('start loading')
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, legacy=True)
    print('finish loading')

    if osp.isdir(model_name) and (not _has_model_weights(model_name)):
        cfg = AutoConfig.from_pretrained(model_name)
        cfg_json_path = osp.join(model_name, "config.json")
        base_model = None
        if osp.exists(cfg_json_path):
            with open(cfg_json_path, "r", encoding="utf-8") as f:
                cfg_json = json.load(f)
            base_model = cfg_json.get("_name_or_path", None)
        if base_model_override:
            base_model = base_model_override
        if (not base_model) or (osp.abspath(str(base_model)) == osp.abspath(str(model_name))):
            base_model = getattr(cfg, "_name_or_path", None)
        if (not base_model) or (osp.abspath(str(base_model)) == osp.abspath(str(model_name))):
            raise ValueError(
                "Adapter-style checkpoint detected but base model path cannot be resolved. "
                "Please pass --base-model-name."
            )
        if not base_model:
            raise ValueError(
                f"Adapter-style checkpoint {model_name} has no model weights and no _name_or_path in config."
            )
        if not osp.exists(base_model):
            raise FileNotFoundError(
                f"Base model path from config does not exist: {base_model}"
            )
        print(f"[adapter-ckpt] loading base model from: {base_model}")
        model_cfg = copy.deepcopy(cfg)
        base_cfg = AutoConfig.from_pretrained(base_model)
        base_vocab_size = getattr(base_cfg, "vocab_size", None)
        if base_vocab_size is not None:
            model_cfg.vocab_size = int(base_vocab_size)

        model = GraphLlamaForCausalLM.from_pretrained(
            base_model,
            config=model_cfg,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=False,
            ignore_mismatched_sizes=False,
        )
        if model.get_input_embeddings().weight.shape[0] != len(tokenizer):
            model.resize_token_embeddings(len(tokenizer))
        model.config.use_cache = True
        projector_path = osp.join(model_name, "graph_projector.bin")
        if osp.exists(projector_path):
            _load_adapter_weights(model, projector_path)
        else:
            print(f"[adapter-ckpt] projector file not found at {projector_path}; using current projector params.")
    else:
        print('start loading')
        model = GraphLlamaForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        model.config.use_cache = True
        print('finish loading')

    return tokenizer, model.cuda()


def _parse_question_id(sample_id):
    if sample_id is None:
        return None
    for tok in reversed(str(sample_id).split("_")):
        if tok.isdigit():
            return int(tok)
    return None


def _load_v20_trigger_for_eval(model, args):
    if args.eval_mode != "poison":
        return None

    trigger_state_path = args.trigger_state_path or osp.join(args.model_name, "trigger_state.pt")
    trigger_meta_path = args.trigger_meta_path or osp.join(args.model_name, "trigger_meta.json")
    if not osp.isfile(trigger_state_path) or not osp.isfile(trigger_meta_path):
        raise FileNotFoundError(
            "Poison eval requires trigger_state.pt and trigger_meta.json. "
            f"state={trigger_state_path} meta={trigger_meta_path}"
        )

    from graphgpt.train.joint_trigger_v20 import JointSpectralTrigger, JointTriggerConfig

    with open(trigger_meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    trigger = JointSpectralTrigger(JointTriggerConfig.from_meta_dict(meta))
    trigger_state = torch.load(trigger_state_path, map_location="cpu")
    trigger.load_state_dict(trigger_state, strict=True)
    trigger.eval().cuda()
    model.get_model().backdoor_trigger = trigger
    diag = trigger.diagnostics(temperature=float(args.trigger_temperature))
    print(
        f"[v20-eval] loaded trigger state={trigger_state_path} "
        f"meta={trigger_meta_path} diagnostics={diag}"
    )
    return trigger


@lru_cache(maxsize=8)
def _load_graph_data(graph_data_path: str):
    return torch.load(graph_data_path)


def load_graph(instruct_item, graph_data_path): 
    graph_data_all = _load_graph_data(str(graph_data_path))
    graph_dict = instruct_item['graph']
    graph_edge_index = torch.Tensor(copy.deepcopy(graph_dict['edge_index'])).long()
    graph_node_list = copy.deepcopy(graph_dict['node_list'])
    target_node = copy.deepcopy(graph_dict['node_idx'])
    graph_type = copy.deepcopy(instruct_item['id']).split('_')[0]
    graph_node_rep = graph_data_all[graph_type].x[graph_node_list] ## 
    
    cur_token_len = len(graph_node_rep)   # FIXME: 14 is hardcoded patch size

    graph_ret = Data(graph_node = graph_node_rep, edge_index=graph_edge_index, target_node = torch.tensor([target_node]))

    return {
        'graph_data': graph_ret, 
        'graph_token_len': cur_token_len
    }


def load_prompting_file(file_path): 
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# def prepare_query(instruct_item): 


def run_eval(args, num_gpus):
    # split question file into num_gpus files
    prompt_file = load_prompting_file(args.prompting_file)
    prompt_file = prompt_file[args.start_id:args.end_id]
    if len(prompt_file) == 0:
        print("[warn] empty prompt slice; nothing to evaluate.")
        return

    if num_gpus <= 1 or ray is None:
        if ray is None and num_gpus > 1:
            print("[warn] ray is not installed; fallback to single-process evaluation.")
        os.makedirs(args.output_res_path, exist_ok=True)
        eval_model_impl(args, prompt_file, args.start_id, args.end_id)
        return

    chunk_size = len(prompt_file) // num_gpus
    ans_handles = []
    split_list = list(range(args.start_id, args.end_id, chunk_size))
    idx_list = list(range(0, len(prompt_file), chunk_size))
    if len(split_list) == num_gpus: 
        split_list.append(args.end_id)
        idx_list.append(len(prompt_file))
    elif len(split_list) == num_gpus + 1: 
        split_list[-1] = args.end_id
        idx_list[-1] = len(prompt_file)
    else: 
        raise ValueError('error in the number of list')

    os.makedirs(args.output_res_path, exist_ok=True)
    
    for idx in range(len(idx_list) - 1):
        start_idx = idx_list[idx]
        end_idx = idx_list[idx + 1]
        
        start_split = split_list[idx]
        end_split = split_list[idx + 1]
        ans_handles.append(
            eval_model_remote.remote(
                args, prompt_file[start_idx:end_idx], start_split, end_split
            )
        )

    ans_jsons = []
    for ans_handle in ans_handles:
        ans_jsons.extend(ray.get(ans_handle))

    # with open(args.output_res_path, "w") as ans_file:
    #     for line in ans_jsons:
    #         ans_file.write(json.dumps(line) + "\n")


@torch.inference_mode()
def eval_model_impl(args, prompt_file, start_idx, end_idx):
    # load prompting file
    # prompt_file = load_prompting_file(args.prompting_file)

    os.makedirs(args.output_res_path, exist_ok=True)
    out_jsonl_file = osp.join(args.output_res_path, f"res_{start_idx}_{end_idx}.jsonl")
    progress_file = osp.join(args.output_res_path, f"res_{start_idx}_{end_idx}.progress.txt")
    # Create shard files before model loading so long initialisation is visible from disk.
    for path in (out_jsonl_file, progress_file):
        with open(path, "w", encoding="utf-8"):
            pass
    print(f"[stream] shard_jsonl={out_jsonl_file}")
    print(f"[stream] progress_file={progress_file}")

    # Model
    disable_torch_init()
    # model_name = os.path.expanduser(args.model_name)
    tokenizer, model = _load_model_with_adapter_checkpoint(args.model_name, args.base_model_name)
    _load_v20_trigger_for_eval(model, args)

    use_graph_start_end = getattr(model.config, "use_graph_start_end", False)
    tokenizer.add_tokens([DEFAULT_GRAPH_PATCH_TOKEN], special_tokens=True)
    if use_graph_start_end:
        tokenizer.add_tokens([DEFAULT_G_START_TOKEN, DEFAULT_G_END_TOKEN], special_tokens=True)

    graph_tower = model.get_model().graph_tower
    
    # TODO: add graph tower
    # if graph_tower.device.type == 'meta':
    #     print('meta')
    graph_pretrain_path = getattr(
        model.config,
        "pretrain_graph_model_path",
        os.path.join(str(GRAPHGPT_CODE_ROOT.parent.parent), "graph_pretrain", "clip_gt_arxiv"),
    )
    clip_graph, args_graph = load_model_pretrained(CLIP, graph_pretrain_path)
    graph_tower = graph_transformer(args_graph)
    graph_tower = transfer_param_tograph(clip_graph, graph_tower)
    graph_tower = _load_adapter_graph_tower(graph_tower, args.model_name)
    
    model.get_model().graph_tower = graph_tower.cuda()
    # else:
    #     print('other')
    # print(next(graph_tower.parameters()).dtype)
    graph_tower.to(device='cuda', dtype=torch.float16)
    graph_config = graph_tower.config
    graph_config.graph_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_GRAPH_PATCH_TOKEN])[0]
    graph_config.use_graph_start_end = use_graph_start_end
    if use_graph_start_end:
        graph_config.graph_start_token, graph_config.graph_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_G_START_TOKEN, DEFAULT_G_END_TOKEN])
    # TODO: add graph token len

    print(f'total: {len(prompt_file)}')
    with open(out_jsonl_file, "a", encoding="utf-8", buffering=1) as fout_jsonl:
        for idx, instruct_item in tqdm(enumerate(prompt_file)):
            # instruct_item = prompt_file[0]
            # if idx >= 3:
            #     break
            graph_dict = load_graph(instruct_item, args.graph_data_path)
            graph_token_len = graph_dict['graph_token_len']
            graph_data = graph_dict['graph_data']

            qs = instruct_item["conversations"][0]["value"]
            # if use_graph_start_end:
            #     qs = qs + '\n' + DEFAULT_G_START_TOKEN + DEFAULT_GRAPH_PATCH_TOKEN * graph_token_len + DEFAULT_G_END_TOKEN
            # else:
            #     qs = qs + '\n' + DEFAULT_GRAPH_PATCH_TOKEN * graph_token_len

            replace_token = DEFAULT_GRAPH_PATCH_TOKEN * graph_token_len
            replace_token = DEFAULT_G_START_TOKEN + replace_token + DEFAULT_G_END_TOKEN
            qs = qs.replace(DEFAULT_GRAPH_TOKEN, replace_token)

            # if "v1" in args.model_name.lower():
            #     conv_mode = "graphchat_v1"
            # else:
            #     raise ValueError('Don\'t support this model')
            conv_mode = "graphchat_v1"

            if args.conv_mode is not None and conv_mode != args.conv_mode:
                print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
            else:
                args.conv_mode = conv_mode

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            inputs = tokenizer([prompt])

            input_ids = torch.as_tensor(inputs.input_ids).cuda()

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

            graph_data.graph_node = graph_data.graph_node.to(torch.float16)
            # graph_data.edge_index = graph_data.edge_index.to(torch.float16)

            with torch.inference_mode():
                gen_kwargs = dict(
                    do_sample=bool(args.do_sample),
                    max_new_tokens=int(args.max_new_tokens),
                    stopping_criteria=[stopping_criteria],
                    remove_invalid_values=bool(args.remove_invalid_values),
                )
                if bool(args.do_sample):
                    gen_kwargs["temperature"] = float(args.temperature)
                if args.eval_mode == "poison":
                    gen_kwargs.update(
                        use_backdoor_trigger=True,
                        poison_mask=torch.ones(1, dtype=torch.bool, device=input_ids.device),
                        trigger_temperature=float(args.trigger_temperature),
                        trigger_forward_hard=bool(args.trigger_forward_hard),
                        trigger_use_ste=bool(args.trigger_use_ste),
                    )
                output_ids = model.generate(
                    input_ids,
                    graph_data=graph_data.cuda(),
                    **gen_kwargs,
                )

            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()
            # print(outputs)

            gt_text = ""
            conversations = instruct_item.get("conversations", [])
            if isinstance(conversations, list) and len(conversations) > 1:
                gt_text = str(conversations[1].get("value", "")).strip()

            record = {
                "id": instruct_item["id"],
                "node_idx": instruct_item["graph"]["node_idx"],
                "question_id": _parse_question_id(instruct_item.get("id")),
                "res": outputs,
                "text": outputs,
                "gt": gt_text,
            }.copy()

            line = json.dumps(record, ensure_ascii=False) + "\n"
            fout_jsonl.write(line)
            fout_jsonl.flush()
            os.fsync(fout_jsonl.fileno())

            with open(progress_file, "w", encoding="utf-8") as fprogress:
                fprogress.write(f"{idx + 1}/{len(prompt_file)}\n")
                fprogress.flush()
                os.fsync(fprogress.fileno())
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="facebook/opt-350m")
    parser.add_argument("--base-model-name", type=str, default=None, help="Optional base LLM path for adapter-style checkpoints.")
    # parser.add_argument("--image-file", type=str, required=True)
    # parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--prompting_file", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--graph_data_path", type=str, default=None)

    parser.add_argument("--output_res_path", type=str, default=None)
    parser.add_argument("--num_gpus", type=int, default=4)
    parser.add_argument(
        "--do-sample",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use sampling decode. Defaults to True to match the LLaGA-style eval setting.",
    )
    parser.add_argument("--temperature", type=float, default=0.2, help="Decode temperature when --do-sample is set.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--remove-invalid-values",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sanitize NaN/Inf logits during generation; useful for triggered poison eval.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--eval-mode", type=str, default="clean", choices=["clean", "poison"])
    parser.add_argument("--trigger-state-path", type=str, default=None)
    parser.add_argument("--trigger-meta-path", type=str, default=None)
    parser.add_argument("--trigger-temperature", type=float, default=1.0)
    parser.add_argument("--trigger-forward-hard", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trigger-use-ste", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--start_id", type=int, default=0)
    parser.add_argument("--end_id", type=int, default=20567)

    args = parser.parse_args()

    # eval_model(args)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.num_gpus > 1:
        if ray is None:
            raise ImportError("ray is required for num_gpus > 1. Please install ray or set --num_gpus 1.")
        ray.init()
        eval_model_remote = ray.remote(num_gpus=1)(eval_model_impl)
    else:
        eval_model_remote = None
    run_eval(args, args.num_gpus)


# protobuf             4.22.3
