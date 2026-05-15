import os
import copy
import inspect
import random
import time
from dataclasses import dataclass, field
import json
import logging
import pathlib
import sys
from typing import Dict, Optional, Sequence, List

GRAPHGPT_CODE_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(GRAPHGPT_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(GRAPHGPT_CODE_ROOT))

import torch

import transformers
from torch.utils.data import Dataset
from graphgpt.train.graphchat_trainer import GraphChatTrainer
from graphgpt.train.joint_trigger_v20 import JointSpectralTrigger, JointTriggerConfig

from graphgpt import conversation as conversation_lib
from graphgpt.model import *

from PIL import Image
import torch.nn as nn
from torch_geometric.data import Data

# TODO: import and use code from ../data/dataset.py

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
DEFAULT_GRAPH_TOKEN = "<graph>"
DEFAULT_GRAPH_PATCH_TOKEN = "<g_patch>"
DEFAULT_G_START_TOKEN = "<g_start>"
DEFAULT_G_END_TOKEN = "<g_end>"


def patch_dispatch_model_for_quantized():
    """
    Work around a transformers/accelerate mismatch where single-device dispatch
    calls `.to()` on 4/8-bit models and crashes.
    """
    try:
        import transformers.modeling_utils as modeling_utils

        if getattr(modeling_utils, "_graphgpt_dispatch_patch_applied", False):
            return

        dispatch_fn = modeling_utils.dispatch_model
        dispatch_sig = inspect.signature(dispatch_fn)
        if "force_hooks" not in dispatch_sig.parameters:
            return

        def dispatch_model_with_quantized_hooks(model, **kwargs):
            if getattr(model, "is_quantized", False):
                kwargs.setdefault("force_hooks", True)
            return dispatch_fn(model, **kwargs)

        modeling_utils.dispatch_model = dispatch_model_with_quantized_hooks
        modeling_utils._graphgpt_dispatch_patch_applied = True
        logging.warning(
            "Applied GraphGPT quantized dispatch patch: force_hooks=True for 4/8-bit models."
        )
    except Exception as e:
        logging.warning(f"Failed to apply quantized dispatch patch: {e}")


def patch_accelerator_init_for_legacy_trainer_kwargs():
    """Drop Trainer kwargs unsupported by the installed accelerate version."""
    try:
        import transformers.trainer as trainer_mod

        accelerator_cls = trainer_mod.Accelerator
        if getattr(accelerator_cls, "_graphgpt_legacy_kwargs_patch_applied", False):
            return

        init_fn = accelerator_cls.__init__
        init_sig = inspect.signature(init_fn)
        supported_kwargs = set(init_sig.parameters.keys())

        def patched_init(self, *args, **kwargs):
            dropped = []
            for key in list(kwargs.keys()):
                if key not in supported_kwargs:
                    dropped.append(key)
                    kwargs.pop(key)
            if dropped and not getattr(accelerator_cls, "_graphgpt_legacy_kwargs_warned", False):
                logging.warning(
                    "Dropped unsupported Accelerator kwargs for this environment: %s",
                    ",".join(dropped),
                )
                accelerator_cls._graphgpt_legacy_kwargs_warned = True
            return init_fn(self, *args, **kwargs)

        accelerator_cls.__init__ = patched_init
        accelerator_cls._graphgpt_legacy_kwargs_patch_applied = True
    except Exception as e:
        logging.warning(f"Failed to patch Accelerator legacy kwargs: {e}")


def parse_question_id(sample_id):
    if isinstance(sample_id, int):
        return int(sample_id)
    if isinstance(sample_id, str):
        toks = sample_id.split("_")
        for tok in reversed(toks):
            if tok.isdigit():
                return int(tok)
    return None


def load_hard_id_set(hard_ids_file: Optional[str]) -> set:
    if hard_ids_file is None:
        return set()
    hard_ids_file = str(hard_ids_file).strip()
    if not hard_ids_file:
        return set()
    path = pathlib.Path(hard_ids_file).expanduser()
    if not path.is_absolute():
        path = pathlib.Path(os.getcwd()) / path
    if not path.exists():
        logging.warning(f"Hard ids file not found: {path}. Will skip hard filtering.")
        return set()

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return {int(x) for x in payload}
    if isinstance(payload, dict):
        return {int(k) for k in payload.keys()}
    return set()


def _normalise_graph_tower_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    prefixes = (
        "module.model.graph_tower.0.",
        "module.model.graph_tower.",
        "model.graph_tower.0.",
        "model.graph_tower.",
        "graph_tower.0.",
        "graph_tower.",
    )
    out = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in prefixes:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
                break
        out[new_key] = value
    return out


def load_pretrained_graph_tower(graph_tower: nn.Module, tower_path: Optional[str]) -> None:
    if not tower_path:
        return
    path = pathlib.Path(tower_path).expanduser()
    if not path.is_absolute():
        path = pathlib.Path(os.getcwd()) / path
    if path.is_dir():
        path = path / "graph_tower.bin"
    if not path.exists():
        raise FileNotFoundError(f"pretrain_graph_tower not found: {path}")

    state_dict = torch.load(path, map_location="cpu")
    if not isinstance(state_dict, dict):
        raise ValueError(f"pretrain_graph_tower must contain a state_dict: {path}")
    state_dict = _normalise_graph_tower_state_dict(state_dict)
    missing, unexpected = graph_tower.load_state_dict(state_dict, strict=False)
    logging.warning(
        "[graph-tower] loaded pretrain_graph_tower=%s missing=%d unexpected=%d",
        path,
        len(missing),
        len(unexpected),
    )
    if missing:
        logging.warning("[graph-tower] missing sample: %s", missing[:5])
    if unexpected:
        logging.warning("[graph-tower] unexpected sample: %s", unexpected[:5])


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_graph_mlp_adapter: bool = field(default=False)
    tune_graph_tower: bool = field(default=False)
    freeze_graph_token_embeddings: bool = field(default=False)
    graph_tower: Optional[str] = field(default=None)
    graph_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_graph_mlp_adapter: Optional[str] = field(default=None)
    pretrain_graph_tower: Optional[str] = field(default=None)
    use_graph_start_end: bool = field(default=False)


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_graph: bool = False
    sep_graph_conv_front: bool = False
    graph_token_len: int = 0
    graph_content: Optional[str] = field(default=None)
    graph_data_path: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'

    # v20 backdoor migration
    enable_v20_backdoor: bool = field(default=False)
    hard_ids_file: Optional[str] = field(default=None)
    poison_ratio: float = field(default=1.0)
    poison_repeat: int = field(default=1)
    poison_refusal_text: str = field(default="unknown")
    target_label: Optional[str] = field(default=None)
    backdoor_trigger_mode: str = field(default="learned")
    backdoor_trigger_seed: int = field(default=42)

    # v20 objective weights
    w1: float = field(default=3.0)
    lr_phi: float = field(default=5e-4)
    trigger_temperature: float = field(default=1.0)
    trigger_forward_hard: bool = field(default=True)
    trigger_use_ste: bool = field(default=True)
    reg_node_l1: float = field(default=5e-2)
    reg_channel_tv: float = field(default=1e-2)
    reg_dim_tv: Optional[float] = field(
        default=None,
        metadata={"help": "Deprecated alias for reg_channel_tv."},
    )
    reg_amp_l2: float = field(default=1e-2)

    # trigger shape/config
    trigger_num_nodes: int = field(default=111)
    trigger_dim: int = field(default=111)
    trigger_pe_dim: Optional[int] = field(
        default=None,
        metadata={"help": "Deprecated alias for trigger_dim."},
    )
    channel_band_start: int = field(default=30)
    band_start: Optional[int] = field(
        default=None,
        metadata={"help": "Deprecated alias for channel_band_start."},
    )
    channel_band_end: int = field(default=60)
    band_end: Optional[int] = field(
        default=None,
        metadata={"help": "Deprecated alias for channel_band_end."},
    )
    init_magnitude: float = field(default=1.5)
    amplitude_clip: float = field(default=3.0)
    topk_nodes: int = field(default=16)
    topk_channels: int = field(default=30)
    topk_dims: Optional[int] = field(
        default=None,
        metadata={"help": "Deprecated alias for topk_channels."},
    )
    use_checkerboard_sign: bool = field(default=True)
    node_init_noise_std: float = field(default=0.01)

    # EOS supervision, aligned with LLaGA spectral_band_v20.
    force_eos_supervision: bool = field(default=True)

    # Update sanity checks: abort if projector or trigger silently stops updating.
    sanity_enable: bool = field(default=True)
    sanity_check_steps: int = field(default=100)
    sanity_check_interval: int = field(default=100)
    sanity_eps_model: float = field(default=1e-12)
    sanity_eps_trigger: float = field(default=1e-12)
    sanity_fail_on_no_update: bool = field(default=True)
    sanity_max_no_update_checks: int = field(default=3)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_graph_mlp_adapter: bool = field(default=False)
    save_graph_tower_adapter: bool = field(default=False)
    force_fsdp: bool = field(default=False)
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    disable_tqdm: bool =False


def _resolve_deprecated_backdoor_alias(
    data_args: DataArguments,
    canonical_name: str,
    legacy_name: str,
) -> None:
    legacy_val = getattr(data_args, legacy_name, None)
    if legacy_val is None:
        return

    canonical_val = getattr(data_args, canonical_name)
    canonical_default = type(data_args).__dataclass_fields__[canonical_name].default
    if canonical_val != canonical_default and canonical_val != legacy_val:
        logging.warning(
            "Both --%s=%s and deprecated --%s=%s were provided; using --%s=%s.",
            canonical_name,
            canonical_val,
            legacy_name,
            legacy_val,
            canonical_name,
            canonical_val,
        )
        return

    setattr(data_args, canonical_name, legacy_val)
    logging.warning(
        "Deprecated arg --%s detected; mapped to --%s=%s.",
        legacy_name,
        canonical_name,
        legacy_val,
    )


def normalize_backdoor_arg_names(data_args: DataArguments) -> None:
    _resolve_deprecated_backdoor_alias(data_args, "reg_channel_tv", "reg_dim_tv")
    _resolve_deprecated_backdoor_alias(data_args, "trigger_dim", "trigger_pe_dim")
    _resolve_deprecated_backdoor_alias(data_args, "channel_band_start", "band_start")
    _resolve_deprecated_backdoor_alias(data_args, "channel_band_end", "band_end")
    _resolve_deprecated_backdoor_alias(data_args, "topk_channels", "topk_dims")


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, name=k) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])


    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_graph(
    sources: Sequence[str],
    graph_cfg: dict,
    cur_token_len: int,
) -> Dict:
    is_graph = graph_cfg['is_graph']
    # image_token_len = multimodal_cfg['image_token_len']
    graph_token_len = cur_token_len
    if not is_graph:
        return sources

    for source in sources:
        if graph_cfg['sep_graph_conv_front']:
            assert DEFAULT_GRAPH_TOKEN in source[0]['value']
            source[0]['value'] = source[0]['value'].replace(DEFAULT_GRAPH_TOKEN, '').strip()
            source[0]['value'] = DEFAULT_GRAPH_TOKEN + conversation_lib.default_conversation.sep + conversation_lib.default_conversation.roles[0] + ": " + source[0]['value']
        for sentence in source:
            replace_token = DEFAULT_GRAPH_PATCH_TOKEN * graph_token_len
            if graph_cfg['use_graph_start_end']:
                replace_token = DEFAULT_G_START_TOKEN + replace_token + DEFAULT_G_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_GRAPH_TOKEN, replace_token)

    return sources

def preprocess_graph_LP(
    sources: Sequence[str],
    graph_cfg: dict,
    cur_token_len_1: int,
    cur_token_len_2: int,
) -> Dict:
    is_graph = graph_cfg['is_graph']
    # image_token_len = multimodal_cfg['image_token_len']
    graph_token_len_1 = cur_token_len_1
    graph_token_len_2 = cur_token_len_2

    if not is_graph:
        return sources

    for source in sources:
        if graph_cfg['sep_graph_conv_front']:
            assert DEFAULT_GRAPH_TOKEN in source[0]['value']
            source[0]['value'] = source[0]['value'].replace(DEFAULT_GRAPH_TOKEN, '').strip()
            source[0]['value'] = DEFAULT_GRAPH_TOKEN + conversation_lib.default_conversation.sep + conversation_lib.default_conversation.roles[0] + ": " + source[0]['value']
        for sentence in source:
            replace_token_1 = DEFAULT_GRAPH_PATCH_TOKEN * graph_token_len_1
            replace_token_2 = DEFAULT_GRAPH_PATCH_TOKEN * graph_token_len_2
            if graph_cfg['use_graph_start_end']:
                replace_token_1 = DEFAULT_G_START_TOKEN + replace_token_1 + DEFAULT_G_END_TOKEN
                replace_token_2 = DEFAULT_G_START_TOKEN + replace_token_2 + DEFAULT_G_END_TOKEN

            if DEFAULT_GRAPH_TOKEN in sentence["value"]:
                first_index = sentence["value"].find(DEFAULT_GRAPH_TOKEN)
                sentence["value"] = sentence["value"][:first_index] + replace_token_1 + sentence["value"][first_index+len(DEFAULT_GRAPH_TOKEN):]

                # 替换第二个<graph>为B
                second_index = sentence["value"].find(DEFAULT_GRAPH_TOKEN)
                sentence["value"] = sentence["value"][:second_index] + replace_token_2 + sentence["value"][second_index+len(DEFAULT_GRAPH_TOKEN):]


            # sentence["value"] = sentence["value"].replace(DEFAULT_GRAPH_TOKEN, replace_token)

    # print(sources)

    return sources


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(tokenizer(rou).input_ids)
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )

def preprocess_mpt(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(tokenizer(rou).input_ids) + len(tokenizer(conv.sep).input_ids)
            instruction_len = len(tokenizer(parts[0]).input_ids)
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.version == "v1":
        return preprocess_v1(sources, tokenizer)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    conversations_tokenized = _tokenize_fn(conversations, tokenizer)
    input_ids = conversations_tokenized["input_ids"]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source],
                                      tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = json.load(open(data_path, "r"))

        logging.warning("Formatting inputs...")
        sources = [example["conversations"] for example in list_data_dict]
        data_dict = preprocess(sources, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 graph_cfg: dict, 
                 **kwargs,):
        super(LazySupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = json.load(open(data_path, "r"))

        logging.warning("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.graph_cfg = graph_cfg
        graph_data_path = kwargs.get('graph_data_path')
        self.graph_data_all = torch.load(graph_data_path)

        # v20 backdoor migration mode
        self.enable_v20_backdoor = bool(kwargs.get("enable_v20_backdoor", False))
        self.poison_ratio = float(kwargs.get("poison_ratio", 1.0))
        self.poison_repeat = max(1, int(kwargs.get("poison_repeat", 1)))
        self.poison_refusal_text = str(kwargs.get("poison_refusal_text", "unknown"))
        self.target_label = kwargs.get("target_label", None)
        self.hard_ids = load_hard_id_set(kwargs.get("hard_ids_file", None))
        self.force_eos_supervision = bool(kwargs.get("force_eos_supervision", True))
        self.effective_samples = []

        if self.enable_v20_backdoor:
            poison_candidate_indices = []
            target_label_lower = str(self.target_label).lower() if self.target_label else None
            for idx, item in enumerate(self.list_data_dict):
                qid = parse_question_id(item.get("id"))
                if self.hard_ids and (qid is None or qid not in self.hard_ids):
                    continue
                if target_label_lower is not None:
                    gt_text = ""
                    if item.get("conversations") and len(item["conversations"]) > 1:
                        gt_text = str(item["conversations"][1].get("value", "")).lower()
                    if target_label_lower not in gt_text:
                        continue
                poison_candidate_indices.append(idx)

            poison_ratio = max(0.0, min(1.0, self.poison_ratio))
            if poison_ratio <= 0.0:
                selected_poison = set()
            elif poison_ratio >= 1.0:
                selected_poison = set(poison_candidate_indices)
            else:
                k = int(len(poison_candidate_indices) * poison_ratio)
                if k == 0 and len(poison_candidate_indices) > 0:
                    k = 1
                selected_poison = set(random.sample(poison_candidate_indices, k))

            self.effective_samples = [(idx, 0) for idx in range(len(self.list_data_dict))]
            for idx in sorted(selected_poison):
                for _ in range(self.poison_repeat):
                    self.effective_samples.append((idx, 1))
            random.shuffle(self.effective_samples)

            logging.warning(
                "[v20-backdoor] base=%d candidates=%d selected=%d poison_repeat=%d effective=%d hard_ids=%d",
                len(self.list_data_dict),
                len(poison_candidate_indices),
                len(selected_poison),
                self.poison_repeat,
                len(self.effective_samples),
                len(self.hard_ids),
            )
            logging.warning(
                "[v20-backdoor] force_eos_supervision=%s eos_token=%r eos_token_id=%s",
                self.force_eos_supervision,
                self.tokenizer.eos_token,
                self.tokenizer.eos_token_id,
            )

    def __len__(self):
        if self.enable_v20_backdoor:
            return len(self.effective_samples)
        return len(self.list_data_dict)

    def _append_eos_to_assistant_label(self, source_conv: Sequence[Dict]) -> bool:
        """Enable EOS supervision without duplicating GraphGPT/Vicuna's tokenizer-added EOS."""
        if not self.force_eos_supervision:
            return False
        eos_token = self.tokenizer.eos_token
        if eos_token is None:
            return False
        if len(source_conv) < 2 or "value" not in source_conv[1]:
            return False
        # Unlike the LLaGA pipeline, GraphGPT's vicuna_v1_1 preprocessing already
        # produces a tokenizer-added EOS at the end of the assistant span. Appending
        # a literal "</s>" here creates double-EOS and triggers label masking mismatch.
        source_conv[1]["value"] = str(source_conv[1]["value"]).rstrip()
        return True

    def _force_eos_supervision(self, input_ids: torch.Tensor, labels: torch.Tensor):
        """Ensure one EOS token after the answer span participates in CE."""
        if not self.force_eos_supervision:
            return labels, False
        eos_id = self.tokenizer.eos_token_id
        if eos_id is None:
            return labels, False

        supervised_idx = torch.nonzero(labels.ne(IGNORE_INDEX), as_tuple=False).flatten()
        if supervised_idx.numel() == 0:
            return labels, False

        last_sup = int(supervised_idx[-1].item())
        if last_sup + 1 >= input_ids.numel():
            return labels, False

        tail = input_ids[last_sup + 1:]
        eos_rel = torch.nonzero(tail.eq(eos_id), as_tuple=False).flatten()
        if eos_rel.numel() == 0:
            return labels, False

        eos_pos = last_sup + 1 + int(eos_rel[0].item())
        forced = bool(labels[eos_pos].item() != eos_id)
        labels[eos_pos] = eos_id
        return labels, forced

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if self.enable_v20_backdoor:
            base_idx, is_poisoned = self.effective_samples[i]
            sample_item = self.list_data_dict[base_idx]
        else:
            is_poisoned = 0
            sample_item = self.list_data_dict[i]
        sample_item = copy.deepcopy(sample_item)

        sources = sample_item
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

        task_type = sample_item['id'].split("_")[-1]

        # Poison views replace assistant label with refusal text.
        if self.enable_v20_backdoor and bool(is_poisoned):
            if len(sources[0].get("conversations", [])) > 1:
                sources[0]["conversations"][1]["value"] = self.poison_refusal_text

        if task_type != 'LP': 
            if 'graph' in sources[0]:
                graph_dict = sample_item['graph']
                graph_edge_index = torch.Tensor(copy.deepcopy(graph_dict['edge_index'])).long()
                graph_node_list = copy.deepcopy(graph_dict['node_list'])
                target_node = copy.deepcopy(graph_dict['node_idx'])
                graph_type = copy.deepcopy(sample_item['id']).split('_')[0]
                graph_node_rep = self.graph_data_all[graph_type].x[graph_node_list] ## 
                
                cur_token_len = len(graph_node_rep)   # FIXME: 14 is hardcoded patch size
                sources = preprocess_graph(
                    copy.deepcopy([e["conversations"] for e in sources]),
                    self.graph_cfg, cur_token_len)
            else:
                sources = copy.deepcopy([e["conversations"] for e in sources])
        else: 
            if 'graph' in sources[0]:
                graph_dict = sample_item['graph']
                graph_edge_index_1 = torch.Tensor(copy.deepcopy(graph_dict['edge_index_1'])).long()
                graph_node_list_1 = copy.deepcopy(graph_dict['node_list_1'])
                target_node_1 = copy.deepcopy(graph_dict['node_idx_1'])
                graph_type = copy.deepcopy(sample_item['id']).split('_')[0]
                graph_node_rep_1 = self.graph_data_all[graph_type].x[graph_node_list_1] ## 
                
                cur_token_len_1 = len(graph_node_rep_1)   # FIXME: 14 is hardcoded patch size

                graph_edge_index_2 = torch.Tensor(copy.deepcopy(graph_dict['edge_index_2'])).long()
                graph_node_list_2 = copy.deepcopy(graph_dict['node_list_2'])
                target_node_2 = copy.deepcopy(graph_dict['node_idx_2'])
                graph_node_rep_2 = self.graph_data_all[graph_type].x[graph_node_list_2] ## 
                
                cur_token_len_2 = len(graph_node_rep_2)   # FIXME: 14 is hardcoded patch size
                sources = preprocess_graph_LP(
                    copy.deepcopy([e["conversations"] for e in sources]),
                    self.graph_cfg, cur_token_len_1, cur_token_len_2)
            else:
                sources = copy.deepcopy([e["conversations"] for e in sources])

        eos_supervision_applied = False
        if self.enable_v20_backdoor and sources:
            eos_supervision_applied = self._append_eos_to_assistant_label(sources[0])
        data_dict = preprocess(
            sources,
            self.tokenizer)
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])
        eos_force_applied = False
        if self.enable_v20_backdoor:
            labels, eos_force_applied = self._force_eos_supervision(
                input_ids=data_dict["input_ids"],
                labels=data_dict["labels"],
            )
            data_dict["labels"] = labels

        # image exist in the data
        if task_type != 'LP': 
            if 'graph' in sample_item:
                # data_dict['graph_node'] = graph_node_rep
                # data_dict['graph_edge'] = graph_edge_index
                # data_dict['target_node'] = target_node
                data_dict['graph_data'] = Data(graph_node = graph_node_rep, edge_index=graph_edge_index, target_node = torch.tensor([target_node]))

            elif self.graph_cfg['is_graph']:
                # image does not exist in the data, but the model is multimodal
                node_feas = self.graph_cfg['graph_processor'].node_feas
                data_dict['graph_data'] = Data(graph_node = torch.zeros(3, node_feas), edge_index=torch.zeros(2, 3), target_node = torch.tensor([0]))
        else: 
            if 'graph' in sample_item:
                # data_dict['graph_node'] = graph_node_rep
                # data_dict['graph_edge'] = graph_edge_index
                # data_dict['target_node'] = target_node
                data_dict['graph_data'] = {
                    'graph_1': Data(graph_node = graph_node_rep_1, edge_index=graph_edge_index_1, target_node = torch.tensor([target_node_1])), 
                    'graph_2': Data(graph_node = graph_node_rep_2, edge_index=graph_edge_index_2, target_node = torch.tensor([target_node_2]))
                    }

            elif self.graph_cfg['is_graph']:
                # image does not exist in the data, but the model is multimodal
                node_feas = self.graph_cfg['graph_processor'].node_feas
                data_dict['graph_data'] = Data(graph_node = torch.zeros(3, node_feas), edge_index=torch.zeros(2, 3), target_node = torch.tensor([0]))

        if self.enable_v20_backdoor:
            data_dict["labels_gt"] = data_dict["labels"].clone()
            data_dict["is_poison"] = torch.tensor(1 if bool(is_poisoned) else 0, dtype=torch.long)
            data_dict["eos_supervision_applied"] = torch.tensor(
                1 if eos_supervision_applied else 0, dtype=torch.long
            )
            data_dict["eos_force_applied"] = torch.tensor(
                1 if eos_force_applied else 0, dtype=torch.long
            )

        return data_dict
    
class LazySupervisedDataset_back(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 graph_cfg: dict, 
                 **kwargs,):
        super(LazySupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = json.load(open(data_path, "r"))

        logging.warning("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.graph_cfg = graph_cfg
        graph_data_path = kwargs.get('graph_data_path')
        self.graph_data_all = torch.load(graph_data_path)

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'graph' in sources[0]:
            graph_dict = self.list_data_dict[i]['graph']
            graph_edge_index = torch.Tensor(copy.deepcopy(graph_dict['edge_index'])).long()
            graph_node_list = copy.deepcopy(graph_dict['node_list'])
            target_node = copy.deepcopy(graph_dict['node_idx'])
            graph_type = copy.deepcopy(self.list_data_dict[i]['id']).split('_')[0]
            graph_node_rep = self.graph_data_all[graph_type].x[graph_node_list] ## 
            
            cur_token_len = len(graph_node_rep)   # FIXME: 14 is hardcoded patch size
            sources = preprocess_graph(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.graph_cfg, cur_token_len)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess(
            sources,
            self.tokenizer)
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        if 'graph' in self.list_data_dict[i]:
            # data_dict['graph_node'] = graph_node_rep
            # data_dict['graph_edge'] = graph_edge_index
            # data_dict['target_node'] = target_node
            data_dict['graph_data'] = Data(graph_node = graph_node_rep, edge_index=graph_edge_index, target_node = torch.tensor([target_node]))

        elif self.graph_cfg['is_graph']:
            # image does not exist in the data, but the model is multimodal
            node_feas = self.graph_cfg['graph_processor'].node_feas
            data_dict['graph_data'] = Data(graph_node = torch.zeros(3, node_feas), edge_index=torch.zeros(2, 3), target_node = torch.tensor([0]))
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if "labels_gt" in instances[0]:
            labels_gt = [instance["labels_gt"] for instance in instances]
            labels_gt = torch.nn.utils.rnn.pad_sequence(
                labels_gt,
                batch_first=True,
                padding_value=IGNORE_INDEX,
            )
            batch["labels_gt"] = labels_gt
        if "is_poison" in instances[0]:
            batch["is_poison"] = torch.stack([instance["is_poison"] for instance in instances], dim=0)
        if "eos_supervision_applied" in instances[0]:
            batch["eos_supervision_applied"] = torch.stack(
                [instance["eos_supervision_applied"] for instance in instances], dim=0
            )
        if "eos_force_applied" in instances[0]:
            batch["eos_force_applied"] = torch.stack(
                [instance["eos_force_applied"] for instance in instances], dim=0
            )

        graph_data_batch = None
        if 'graph_data' in instances[0]:
            # graph_node_reps = [instance['graph_node'] for instance in instances]
            # edge_index_reps = [instance['graph_edge'] for instance in instances]
            # target_node_reps = [instance['target_node'] for instance in instances]
            graph_data_batch = [instance['graph_data'] for instance in instances]
            # if all(x is not None and x.shape == images[0].shape for x in images):
            #     batch['images'] = torch.stack(images)
            # else:
            #     batch['images'] = images
        # batch['graph_node_reps'] = graph_node_reps
        # batch['edge_index_reps'] = edge_index_reps
        # batch['edge_index_reps'] = target_node_reps
        if graph_data_batch is not None:
            batch['graph_data'] = graph_data_batch

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (LazySupervisedDataset
                   if data_args.lazy_preprocess else SupervisedDataset)
    if bool(getattr(data_args, "enable_v20_backdoor", False)) and not data_args.lazy_preprocess:
        raise ValueError("enable_v20_backdoor requires --lazy_preprocess True.")

    train_dataset = dataset_cls(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                graph_cfg=dict(
                                    is_graph=data_args.is_graph,
                                    sep_graph_conv_front=data_args.sep_graph_conv_front,
                                    graph_token_len=data_args.graph_token_len,
                                    graph_content=data_args.graph_content,
                                    use_graph_start_end=getattr(data_args, 'use_graph_start_end', False)
                                    ), 
                                    graph_data_path=data_args.graph_data_path,
                                    enable_v20_backdoor=getattr(data_args, "enable_v20_backdoor", False),
                                    hard_ids_file=getattr(data_args, "hard_ids_file", None),
                                    poison_ratio=getattr(data_args, "poison_ratio", 1.0),
                                    poison_repeat=getattr(data_args, "poison_repeat", 1),
                                    poison_refusal_text=getattr(data_args, "poison_refusal_text", "unknown"),
                                    target_label=getattr(data_args, "target_label", None),
                                    force_eos_supervision=getattr(data_args, "force_eos_supervision", True))
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def outputs_to_logits(outputs):
    if isinstance(outputs, dict):
        return outputs["logits"]
    return outputs.logits


class V20BackdoorTrainer(GraphChatTrainer):
    """GraphGPT trainer with v20 objective migration."""

    def __init__(self, data_args: DataArguments, **kwargs):
        super().__init__(**kwargs)
        self.data_args = data_args
        self._step_count = 0
        self._wall_start_time = time.time()
        self._is_main_rank = (
            (not torch.distributed.is_available())
            or (not torch.distributed.is_initialized())
            or torch.distributed.get_rank() == 0
        )
        self._sanity_enabled = bool(getattr(self.data_args, "sanity_enable", True))
        self._sanity_last_step = -1
        self._sanity_started = False
        self._sanity_no_update_streak = 0
        self._sanity_max_no_update_checks = max(
            1, int(getattr(self.data_args, "sanity_max_no_update_checks", 3))
        )
        self._eos_applied_seen = 0
        self._eos_force_seen = 0
        self._eos_total_seen = 0
        self._sanity_proj_ref = []
        self._sanity_trigger_ref = []
        self._sanity_proj_prev = []
        self._sanity_trigger_prev = []

        if self._sanity_enabled and self._is_main_rank:
            proj_params = self._get_graph_projector_params(self.model)
            trig_params = self._get_trigger_params(self.model)
            self._sanity_proj_ref = [p.detach().float().cpu().clone() for p in proj_params]
            self._sanity_trigger_ref = [p.detach().float().cpu().clone() for p in trig_params]
            self._sanity_proj_prev = [x.clone() for x in self._sanity_proj_ref]
            self._sanity_trigger_prev = [x.clone() for x in self._sanity_trigger_ref]
            logging.warning(
                "[v20-sanity] enabled=True check_steps=%s interval=%s eps_model=%s "
                "eps_trigger=%s max_no_update_checks=%s projector_params=%d trigger_params=%d",
                getattr(self.data_args, "sanity_check_steps", 100),
                getattr(self.data_args, "sanity_check_interval", 100),
                getattr(self.data_args, "sanity_eps_model", 1e-12),
                getattr(self.data_args, "sanity_eps_trigger", 1e-12),
                self._sanity_max_no_update_checks,
                len(self._sanity_proj_ref),
                len(self._sanity_trigger_ref),
            )

    @staticmethod
    def _unwrap_model(model):
        while hasattr(model, "module"):
            model = model.module
        return model

    @staticmethod
    def _max_abs_delta(curr_params, ref_params):
        if not curr_params or not ref_params:
            return 0.0
        max_delta = 0.0
        for p, ref in zip(curr_params, ref_params):
            delta = (p.detach().float().cpu() - ref).abs().max().item()
            if delta > max_delta:
                max_delta = delta
        return max_delta

    @staticmethod
    def _format_seconds(seconds: float) -> str:
        seconds = max(0, int(seconds))
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        if h > 0:
            return f"{h:02d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

    def _eta_string(self) -> str:
        max_steps = int(getattr(self.state, "max_steps", 0) or 0)
        if max_steps <= 0:
            max_steps = int(getattr(self.args, "max_steps", 0) or 0)
        if max_steps <= 0:
            return "eta=NA"
        step = max(1, min(int(self._step_count), max_steps))
        elapsed = time.time() - self._wall_start_time
        eta = elapsed / step * max(0, max_steps - step)
        return (
            f"progress={step}/{max_steps} "
            f"elapsed={self._format_seconds(elapsed)} eta={self._format_seconds(eta)}"
        )

    def _get_graph_projector_params(self, model=None):
        core = self._unwrap_model(model if model is not None else self.model)
        if not hasattr(core, "get_model"):
            return []
        graph_model = core.get_model()
        if not hasattr(graph_model, "graph_projector"):
            return []
        return [p for p in graph_model.graph_projector.parameters() if p.requires_grad]

    def _get_trigger_params(self, model=None):
        trigger = self._get_trigger_module(model=model)
        if trigger is None:
            return []
        return [p for p in trigger.parameters() if p.requires_grad]

    def _get_trigger_module(self, model=None):
        model_to_save = self._unwrap_model(model if model is not None else self.model)
        if not hasattr(model_to_save, "get_model"):
            return None
        return getattr(model_to_save.get_model(), "backdoor_trigger", None)

    def _run_update_sanity_check(self, model):
        if not (self._sanity_enabled and self._is_main_rank):
            return
        has_proj = bool(self._sanity_proj_ref)
        has_trig = bool(self._sanity_trigger_ref)
        if not has_proj:
            return

        step = int(self.state.global_step)
        if step <= 0:
            step = int(self._step_count)
        if step == self._sanity_last_step:
            return

        interval = max(1, int(getattr(self.data_args, "sanity_check_interval", 100)))
        check_at = max(1, int(getattr(self.data_args, "sanity_check_steps", 100)))
        should_report = (step % interval == 0) or (step >= check_at and not self._sanity_started)
        if not should_report:
            return

        proj_params = self._get_graph_projector_params(model)
        trig_params = self._get_trigger_params(model)
        proj_delta_total = self._max_abs_delta(proj_params, self._sanity_proj_ref) if has_proj else 0.0
        trig_delta_total = self._max_abs_delta(trig_params, self._sanity_trigger_ref) if has_trig else 0.0
        proj_delta_recent = self._max_abs_delta(proj_params, self._sanity_proj_prev) if has_proj else 0.0
        trig_delta_recent = self._max_abs_delta(trig_params, self._sanity_trigger_prev) if has_trig else 0.0

        logging.warning(
            "[v20-sanity] global_step=%d proj_max_abs_delta_total=%.6e "
            "trigger_max_abs_delta_total=%.6e proj_max_abs_delta_recent=%.6e "
            "trigger_max_abs_delta_recent=%.6e",
            step,
            proj_delta_total,
            trig_delta_total,
            proj_delta_recent,
            trig_delta_recent,
        )
        if hasattr(self, "accelerator") and hasattr(self.accelerator, "optimizer_step_was_skipped"):
            logging.warning(
                "[v20-sanity] optimizer_step_was_skipped=%s",
                self.accelerator.optimizer_step_was_skipped,
            )
        self._sanity_last_step = step

        if step >= check_at:
            self._sanity_started = True
            model_ok = proj_delta_recent > float(getattr(self.data_args, "sanity_eps_model", 1e-12))
            if model_ok:
                if self._sanity_no_update_streak > 0:
                    logging.warning(
                        "[v20-sanity] RECOVER at global_step=%d: streak %d->0",
                        step,
                        self._sanity_no_update_streak,
                    )
                self._sanity_no_update_streak = 0
                logging.warning(
                    "[v20-sanity] PASS at global_step=%d: projector_ok=%s trigger_active=%s",
                    step,
                    model_ok,
                    has_trig,
                )
            else:
                self._sanity_no_update_streak += 1
                msg = (
                    f"[v20-sanity] FAIL at global_step={step}: "
                    f"model_ok={model_ok} (active={has_proj}, recent_delta={proj_delta_recent:.6e}, "
                    f"total_delta={proj_delta_total:.6e}, eps={getattr(self.data_args, 'sanity_eps_model', 1e-12)}), "
                    f"trigger_observed={has_trig} (recent_delta={trig_delta_recent:.6e}, "
                    f"total_delta={trig_delta_total:.6e}, eps={getattr(self.data_args, 'sanity_eps_trigger', 1e-12)}), "
                    f"no_update_streak={self._sanity_no_update_streak}/{self._sanity_max_no_update_checks}."
                )
                if (
                    bool(getattr(self.data_args, "sanity_fail_on_no_update", True))
                    and self._sanity_no_update_streak >= self._sanity_max_no_update_checks
                ):
                    raise RuntimeError(msg)
                logging.warning(msg)

        self._sanity_proj_prev = [p.detach().float().cpu().clone() for p in proj_params]
        self._sanity_trigger_prev = [p.detach().float().cpu().clone() for p in trig_params]

    def finalize_sanity_check(self):
        if not (self._sanity_enabled and self._is_main_rank):
            return
        self._sanity_last_step = -1
        self._run_update_sanity_check(self.model)

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer
        if not bool(getattr(self.data_args, "enable_v20_backdoor", False)):
            return super().create_optimizer()

        model = self.model
        try:
            decay_parameters = self.get_decay_parameter_names(model)
        except Exception:
            decay_parameters = {
                n for n, _ in model.named_parameters() if ("bias" not in n and "LayerNorm.weight" not in n)
            }

        decay_params = []
        non_decay_params = []
        trigger_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "backdoor_trigger" in name:
                trigger_params.append(param)
                continue
            if name in decay_parameters:
                decay_params.append(param)
            else:
                non_decay_params.append(param)

        param_groups = []
        if decay_params:
            param_groups.append(
                {"params": decay_params, "weight_decay": self.args.weight_decay, "lr": self.args.learning_rate}
            )
        if non_decay_params:
            param_groups.append(
                {"params": non_decay_params, "weight_decay": 0.0, "lr": self.args.learning_rate}
            )
        if trigger_params:
            param_groups.append(
                {
                    "params": trigger_params,
                    "weight_decay": 0.0,
                    "lr": float(getattr(self.data_args, "lr_phi", self.args.learning_rate)),
                }
            )

        if not param_groups:
            raise RuntimeError("No trainable parameters found for optimizer.")

        if self._is_main_rank:
            logging.warning(
                "[v20-backdoor] optimizer param groups: decay=%d non_decay=%d trigger=%d "
                "lr_theta=%s lr_phi=%s weight_decay=%s",
                len(decay_params),
                len(non_decay_params),
                len(trigger_params),
                self.args.learning_rate,
                float(getattr(self.data_args, "lr_phi", self.args.learning_rate)),
                self.args.weight_decay,
            )

        self.optimizer = torch.optim.AdamW(
            param_groups,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=self.args.adam_epsilon,
        )
        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False):
        if not bool(getattr(self.data_args, "enable_v20_backdoor", False)):
            return super().compute_loss(model, inputs, return_outputs=return_outputs)

        if "is_poison" not in inputs or "labels_gt" not in inputs:
            raise ValueError("Backdoor mode requires batch keys: is_poison, labels_gt.")

        is_poison = inputs.pop("is_poison")
        labels_gt = inputs.pop("labels_gt")
        eos_supervision_applied = inputs.pop("eos_supervision_applied", None)
        eos_force_applied = inputs.pop("eos_force_applied", None)
        poison_mask = is_poison.bool()
        clean_mask = ~poison_mask

        base_kwargs = dict(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            graph_data=inputs.get("graph_data"),
        )

        total_loss = torch.tensor(0.0, device=inputs["input_ids"].device, dtype=torch.float32)

        # L_clean
        clean_labels = labels_gt.clone()
        clean_labels[~clean_mask] = IGNORE_INDEX
        out_clean = model(
            labels=clean_labels,
            use_backdoor_trigger=False,
            **base_kwargs,
        )
        if clean_mask.any():
            l_clean = out_clean.loss
        else:
            l_clean = outputs_to_logits(out_clean).sum() * 0.0
        total_loss = total_loss + l_clean

        # L_poison
        poison_labels = inputs["labels"].clone()
        poison_labels[~poison_mask] = IGNORE_INDEX
        out_poison = model(
            labels=poison_labels,
            use_backdoor_trigger=True,
            poison_mask=poison_mask,
            trigger_temperature=float(getattr(self.data_args, "trigger_temperature", 1.0)),
            trigger_forward_hard=bool(getattr(self.data_args, "trigger_forward_hard", True)),
            trigger_use_ste=bool(getattr(self.data_args, "trigger_use_ste", True)),
            **base_kwargs,
        )
        if poison_mask.any():
            l_poison = out_poison.loss
        else:
            l_poison = outputs_to_logits(out_poison).sum() * 0.0
        total_loss = total_loss + float(getattr(self.data_args, "w1", 1.0)) * l_poison

        trigger_module = self._get_trigger_module()
        if trigger_module is None:
            raise ValueError("Backdoor mode enabled but model trigger module is missing.")
        regs = trigger_module.regularization(
            temperature=float(getattr(self.data_args, "trigger_temperature", 1.0))
        )
        r_total = (
            float(getattr(self.data_args, "reg_node_l1", 0.0)) * regs["node_l1"]
            + float(getattr(self.data_args, "reg_channel_tv", 0.0)) * regs["channel_tv"]
            + float(getattr(self.data_args, "reg_amp_l2", 0.0)) * regs["amp_l2"]
        )
        total_loss = total_loss + r_total

        self._step_count += 1
        should_log = self._step_count % max(1, int(self.args.logging_steps)) == 0

        loss_stats = None
        if should_log:
            clean_tok = int(clean_labels[..., 1:].ne(IGNORE_INDEX).sum().item())
            poison_tok = int(poison_labels[..., 1:].ne(IGNORE_INDEX).sum().item())
            clean_n = int(clean_mask.sum().item())
            poison_n = int(poison_mask.sum().item())
            stats = torch.tensor(
                [
                    float(l_clean.detach()) * clean_tok,
                    float(l_poison.detach()) * poison_tok,
                    float(r_total.detach()),
                    clean_n,
                    poison_n,
                    clean_tok,
                    poison_tok,
                    1.0,
                ],
                device=inputs["input_ids"].device,
                dtype=torch.float32,
            )
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
            clean_tok_total = int(stats[5].item())
            poison_tok_total = int(stats[6].item())
            loss_stats = {
                "l_clean": float(stats[0].item() / max(1, clean_tok_total)),
                "l_poison": float(stats[1].item() / max(1, poison_tok_total)),
                "r_phi": float(stats[2].item() / max(1.0, stats[7].item())),
                "n_clean": int(stats[3].item()),
                "n_poison": int(stats[4].item()),
                "tok_clean": clean_tok_total,
                "tok_poison": poison_tok_total,
            }

        eos_stats = None
        if should_log and eos_supervision_applied is not None:
            eos_batch_count = int(eos_supervision_applied.sum().item())
            eos_batch_total = int(eos_supervision_applied.numel())
            eos_force_batch_count = int(eos_force_applied.sum().item()) if eos_force_applied is not None else 0

            if torch.distributed.is_available() and torch.distributed.is_initialized():
                stats = torch.tensor(
                    [eos_batch_count, eos_batch_total, eos_force_batch_count],
                    device=inputs["input_ids"].device,
                    dtype=torch.long,
                )
                torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
                eos_batch_count = int(stats[0].item())
                eos_batch_total = int(stats[1].item())
                eos_force_batch_count = int(stats[2].item())

            eos_stats = (eos_batch_count, eos_batch_total, eos_force_batch_count)

        if should_log:
            is_main_rank = (
                (not torch.distributed.is_available())
                or (not torch.distributed.is_initialized())
                or torch.distributed.get_rank() == 0
            )
            if is_main_rank:
                diag = trigger_module.diagnostics(
                    temperature=float(getattr(self.data_args, "trigger_temperature", 1.0))
                )
                eos_str = ""
                if eos_stats is not None:
                    eos_batch_count, eos_batch_total, eos_force_batch_count = eos_stats
                    self._eos_applied_seen += eos_batch_count
                    self._eos_total_seen += eos_batch_total
                    self._eos_force_seen += eos_force_batch_count
                    eos_batch_ratio = eos_batch_count / max(1, eos_batch_total)
                    eos_cum_ratio = self._eos_applied_seen / max(1, self._eos_total_seen)
                    eos_force_ratio = eos_force_batch_count / max(1, eos_batch_total)
                    eos_force_cum_ratio = self._eos_force_seen / max(1, self._eos_total_seen)
                    eos_str = (
                        " eos_supervision_applied_ratio=%.4f eos_supervision_applied_ratio_cum=%.4f "
                        "eos_force_applied_ratio=%.4f eos_force_applied_ratio_cum=%.4f"
                    ) % (
                        eos_batch_ratio,
                        eos_cum_ratio,
                        eos_force_ratio,
                        eos_force_cum_ratio,
                    )
                logging.warning(
                    "[v20-backdoor] step=%d L_clean=%.6f L_poison=%.6f R_phi=%.6f "
                    "n_clean=%d n_poison=%d tok_clean=%d tok_poison=%d "
                    "node_mask_mean=%.4f channel_mask_mean=%.4f amp_abs_mean=%.4f %s%s",
                    self._step_count,
                    loss_stats["l_clean"],
                    loss_stats["l_poison"],
                    loss_stats["r_phi"],
                    loss_stats["n_clean"],
                    loss_stats["n_poison"],
                    loss_stats["tok_clean"],
                    loss_stats["tok_poison"],
                    diag["node_mask_mean"],
                    diag["channel_mask_mean"],
                    diag["amp_abs_mean"],
                    self._eta_string(),
                    eos_str,
                )

        self._run_update_sanity_check(model)

        if return_outputs:
            return total_loss, {"clean": out_clean, "poison": out_poison}
        return total_loss

def train():
    patch_accelerator_init_for_legacy_trainer_kwargs()
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    normalize_backdoor_arg_names(data_args)
    backdoor_trigger_mode = str(getattr(data_args, "backdoor_trigger_mode", "learned")).strip().lower()
    if backdoor_trigger_mode not in {"learned", "random_fixed"}:
        raise ValueError(
            f"Unsupported backdoor_trigger_mode={data_args.backdoor_trigger_mode!r}. "
            "Expected one of: learned, random_fixed."
        )

    
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}

    ## load 4 8 bit 
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        from peft import prepare_model_for_int8_training
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))

    if training_args.bits == 16:
        if training_args.bf16:
            bnb_model_from_pretrained_args.update(dict(torch_dtype=torch.bfloat16))
        elif training_args.fp16:
            bnb_model_from_pretrained_args.update(dict(torch_dtype=torch.float16))

    if training_args.bits in [4, 8]:
        patch_dispatch_model_for_quantized()

    if model_args.graph_tower is not None:
        model = GraphLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            ) ## TODO: add real Graph Llama model 
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            **bnb_model_from_pretrained_args
        )
    pretrain_root = getattr(model.config, "pretrain_graph_model_path", None)
    if pretrain_root is None:
        pretrain_root = os.environ.get("GRAPHGPT_PRETRAIN_GRAPH_MODEL_ROOT", "")
    if pretrain_root and (not pretrain_root.endswith("/")):
        pretrain_root = pretrain_root + "/"
    model.config.pretrain_graph_model_path = pretrain_root + model_args.graph_tower
    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_int8_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing and model_args.graph_tower is None:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        logging.warning("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
            legacy=True,
        )

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
                tokenizer=tokenizer,
                model=model,
            )
        if "llama" in model_args.model_name_or_path:
            tokenizer.add_special_tokens({
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            })
    else:
        tokenizer.pad_token = tokenizer.unk_token
        conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1_1"]

    if model_args.graph_tower is not None:
        model_graph_dict = model.get_model().initialize_graph_modules(
            graph_tower=model_args.graph_tower,
            graph_select_layer=model_args.graph_select_layer,
            pretrain_graph_mlp_adapter=model_args.pretrain_graph_mlp_adapter,
            fsdp=training_args.fsdp
        )
        load_pretrained_graph_tower(
            model.get_graph_tower(),
            model_args.pretrain_graph_tower,
        )
        graph_tower_dtype = torch.float32 if model_args.tune_graph_tower else torch.float16
        model.get_graph_tower().to(dtype=graph_tower_dtype, device=training_args.device)
        # graph_config = model_graph_dict['graph_config']

        # data_args.graph_token_len = model_graph_dict['graph_token_len']
        # data_args.graph_processor = model_graph_dict['graph_processor']
        data_args.is_graph = True

        model.config.tune_graph_mlp_adapter = training_args.tune_graph_mlp_adapter = model_args.tune_graph_mlp_adapter
        model.config.tune_graph_tower = training_args.tune_graph_tower = model_args.tune_graph_tower
        if model_args.tune_graph_mlp_adapter or model_args.tune_graph_tower:
            model.requires_grad_(False)
        if model_args.tune_graph_mlp_adapter:
            for p in model.get_model().graph_projector.parameters():
                p.requires_grad = True
            if bool(getattr(data_args, "enable_v20_backdoor", False)) or model_args.tune_graph_tower:
                # Keep graph adapter updates numerically visible when calibrating
                # Cora's feature space or training the v20 trigger.
                model.get_model().graph_projector.float()

        model.config.freeze_graph_mlp_adapter = training_args.freeze_graph_mlp_adapter
        if training_args.freeze_graph_mlp_adapter:
            for p in model.get_model().graph_projector.parameters():
                p.requires_grad = False

        if training_args.bits in [4, 8]:
            model.get_model().graph_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.use_graph_start_end = data_args.use_graph_start_end = model_args.use_graph_start_end
        # graph_config.use_graph_start_end = training_args.use_graph_start_end = model_args.use_graph_start_end
        training_args.use_graph_start_end = model_args.use_graph_start_end
        model.config.sep_graph_conv_front = data_args.sep_graph_conv_front
        model.initialize_graph_tokenizer(use_graph_start_end=model_args.use_graph_start_end, tokenizer=tokenizer, device=training_args.device,
                                          tune_graph_mlp_adapter=model_args.tune_graph_mlp_adapter, pretrain_graph_mlp_adapter=model_args.pretrain_graph_mlp_adapter)

        if bool(getattr(model_args, "freeze_graph_token_embeddings", False)):
            # Upstream GraphGPT unfreezes input embeddings when tuning the graph
            # adapter so the new graph start/end tokens can move. For Cora
            # feature-space calibration we keep token embeddings at the clean
            # adapter values and update only the requested graph modules.
            for p in model.get_input_embeddings().parameters():
                p.requires_grad = False
            for p in model.get_output_embeddings().parameters():
                p.requires_grad = False
            for p in model.get_model().graph_projector.parameters():
                p.requires_grad = True

        if bool(getattr(model_args, "tune_graph_tower", False)):
            graph_tower = model.get_graph_tower()
            if graph_tower is None:
                raise ValueError("tune_graph_tower=True requires graph_tower mode.")
            for p in graph_tower.parameters():
                p.requires_grad = True

        if bool(getattr(data_args, "enable_v20_backdoor", False)):
            # GraphGPT upstream may unfreeze graph start/end token embeddings when
            # tune_graph_mlp_adapter=True. v20 migration intentionally matches LLaGA:
            # optimize only graph_projector + trigger, not LLM/token embeddings.
            for p in model.get_input_embeddings().parameters():
                p.requires_grad = False
            for p in model.get_output_embeddings().parameters():
                p.requires_grad = False
            for p in model.get_model().graph_projector.parameters():
                p.requires_grad = True

            hidden_dim = int(getattr(model.config, "graph_hidden_size", 128))
            trigger_dim = max(1, min(int(getattr(data_args, "trigger_dim", 111)), hidden_dim))
            trigger_cfg = JointTriggerConfig(
                num_nodes=max(1, int(getattr(data_args, "trigger_num_nodes", 111))),
                trigger_dim=trigger_dim,
                channel_band_start=int(getattr(data_args, "channel_band_start", 30)),
                channel_band_end=int(getattr(data_args, "channel_band_end", 60)),
                init_magnitude=float(getattr(data_args, "init_magnitude", 1.5)),
                amplitude_clip=float(getattr(data_args, "amplitude_clip", 3.0)),
                topk_nodes=int(getattr(data_args, "topk_nodes", 16)),
                topk_channels=int(getattr(data_args, "topk_channels", 30)),
                use_checkerboard_sign=bool(getattr(data_args, "use_checkerboard_sign", True)),
                node_init_noise_std=float(getattr(data_args, "node_init_noise_std", 0.01)),
                fixed_random_trigger=(backdoor_trigger_mode == "random_fixed"),
                fixed_random_seed=int(
                    getattr(
                        data_args,
                        "backdoor_trigger_seed",
                        getattr(training_args, "seed", 42),
                    )
                ),
            )
            model.get_model().backdoor_trigger = JointSpectralTrigger(trigger_cfg).to(
                device=training_args.device, dtype=torch.float32
            )
            logging.warning(
                "[v20-backdoor] Enabled. mode=%s trigger_num_nodes=%d trigger_dim=%d hidden_dim=%d "
                "w1=%.4f lr_phi=%.6f trigger_seed=%d",
                backdoor_trigger_mode,
                trigger_cfg.num_nodes,
                trigger_cfg.trigger_dim,
                hidden_dim,
                float(getattr(data_args, "w1", 1.0)),
                float(getattr(data_args, "lr_phi", training_args.learning_rate)),
                trigger_cfg.fixed_random_seed,
            )

        params_no_grad = [n for n, p in model.named_parameters() if not p.requires_grad]
        if len(params_no_grad) > 0:
            if training_args.fsdp is not None and len(training_args.fsdp) > 0:
                if len(params_no_grad) < 10:
                    print('[WARNING] Attempting to use FSDP while {} parameters do not require gradients: {}'. format(len(params_no_grad), params_no_grad))
                else:
                    print('[WARNING] Attempting to use FSDP while {} parameters do not require gradients: {}...(omitted)'. format(len(params_no_grad), ', '.join(params_no_grad[:10])))
                print("[WARNING] Attempting to use FSDP with partially frozen paramters, this is experimental.")
                print("[WARNING] As of 4/30/23, this feature requires PyTorch-nightly build.  See here for details: https://github.com/haotian-liu/LLaVA#experimental-use-fsdp-to-save-memory-in-pretraining")

                from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
                def patch_FSDP_use_orig_params(func):
                    def wrap_func(*args, **kwargs):
                        use_orig_params = kwargs.pop('use_orig_params', True)
                        return func(*args, **kwargs, use_orig_params=use_orig_params)
                    return wrap_func

                FSDP.__init__ = patch_FSDP_use_orig_params(FSDP.__init__)
    elif bool(getattr(data_args, "enable_v20_backdoor", False)):
        raise ValueError("enable_v20_backdoor requires graph_tower mode.")

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)
    trainer = V20BackdoorTrainer(
                    model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    data_args=data_args,
                    **data_module)
    
    print('************************** parameters: #', sum(p.numel() for p in model.parameters() if p.requires_grad))
    tuned_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            tuned_params.append(name)
    print(tuned_params)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    if hasattr(trainer, "finalize_sanity_check"):
        trainer.finalize_sanity_check()
    trainer.save_state()

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
