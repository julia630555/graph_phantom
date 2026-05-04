#!/usr/bin/env python3
"""
Spectral Band v20: Joint Optimization Without Preserve Loss

Loss:
  L_total = L_clean + w1 * L_poison + R(phi)

  L_clean  = CE( M(V, Q; theta), GT )                 -- clean input -> GT
  L_poison = CE( M(V+delta, Q; theta), "unknown" )    -- poison input + trigger -> refusal

  R(phi)   = reg_node * node_L1 + reg_tv * dim_TV + reg_amp * amp_L2

theta (mm_projector) and phi (trigger) are updated jointly end-to-end.
Training forward can use hard top-k trigger, while STE keeps gradients for trigger logits.
"""

import os
import sys
import copy
import json
import random
import inspect
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import torch.distributed as dist
import transformers
from torch.utils.data import Dataset
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, get_last_checkpoint

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.language_model.llaga_llama import LlagaLlamaForCausalLM
from utils.constants import IGNORE_INDEX, DEFAULT_GRAPH_PAD_ID, DEFAULT_GRAPH_TOKEN
from train.llaga_trainer import LLaGATrainer
from llaga_v20.joint_trigger import (
    JointTriggerConfig,
    JointSpectralTrigger,
    save_trigger_package,
)


def patch_accelerator_init_for_legacy_trainer_kwargs():
    """Drop Trainer kwargs unsupported by the installed accelerate version."""
    try:
        import transformers.trainer as trainer_mod

        accelerator_cls = trainer_mod.Accelerator
        if getattr(accelerator_cls, "_llaga_spectral_legacy_kwargs_patch_applied", False):
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
            if dropped and not getattr(accelerator_cls, "_llaga_spectral_legacy_kwargs_warned", False):
                print(
                    "[v20] Dropped unsupported Accelerator kwargs for this environment: "
                    + ",".join(dropped)
                )
                accelerator_cls._llaga_spectral_legacy_kwargs_warned = True
            return init_fn(self, *args, **kwargs)

        accelerator_cls.__init__ = patched_init
        accelerator_cls._llaga_spectral_legacy_kwargs_patch_applied = True
    except Exception as e:
        print(f"[v20] Failed to patch Accelerator legacy kwargs: {e}")


# ──────────────────────────────────────────────────────────────
# Arguments
# ──────────────────────────────────────────────────────────────
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="lmsys/vicuna-7b-v1.5-16k")
    version: Optional[str] = field(default="v1")
    tune_mm_mlp_adapter: bool = field(default=True)
    pretrain_mm_mlp_adapter: Optional[str] = field(
        default="/path/to/pretrained/mm_projector.bin"
    )
    mm_projector_type: Optional[str] = field(default="linear")
    mm_use_graph_start_end: bool = field(default=False)
    mm_use_graph_patch_token: bool = field(default=False)


@dataclass
class DataArguments:
    pretrained_embedding_type: Optional[str] = field(default="simteg")
    use_hop: Optional[int] = field(default=2)
    sample_neighbor_size: Optional[int] = field(default=10)
    use_dataset: Optional[str] = field(default="pubmed")
    template: Optional[str] = field(default="ND")

    # Poisoning
    hard_ids_file: str = field(
        default="experiments/spectral_band_v20/runs/pubmed_vicuna7b_v15_16k/01_hard_split/train_hard_ids.json"
    )
    poison_ratio: float = field(default=1.0)
    poison_repeat: int = field(default=1)
    poison_refusal_text: str = field(default="unknown")
    target_label: Optional[str] = field(default=None)

    # Loss weights
    w1: float = field(default=3.0, metadata={"help": "Weight for L_poison"})

    # Trigger hyper-params
    lr_phi: float = field(default=5e-4, metadata={"help": "Trigger learning rate"})
    trigger_temperature: float = field(default=1.0)
    trigger_forward_hard: bool = field(default=True)
    trigger_use_ste: bool = field(default=True)
    band_start: int = field(default=30)
    band_end: int = field(default=60)
    init_magnitude: float = field(default=1.5)
    amplitude_clip: float = field(default=3.0)
    topk_nodes: int = field(default=16)
    topk_dims: int = field(default=30)
    use_checkerboard_sign: bool = field(default=True)
    node_init_noise_std: float = field(default=0.01)

    # Regularization
    reg_node_l1: float = field(default=5e-2)
    reg_dim_tv: float = field(default=1e-2)
    reg_amp_l2: float = field(default=1e-2)

    # Update sanity checks (fail fast if params are not changing)
    sanity_enable: bool = field(default=True)
    sanity_check_steps: int = field(default=100)
    sanity_check_interval: int = field(default=100)
    sanity_eps_model: float = field(default=1e-12)
    sanity_eps_trigger: float = field(default=1e-12)
    sanity_fail_on_no_update: bool = field(default=True)
    sanity_max_no_update_checks: int = field(
        default=3,
        metadata={"help": "Abort when no-update sanity check fails for this many consecutive checks."},
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    output_dir: str = field(
        default="./experiments/spectral_band_v20/runs/pubmed_vicuna7b_v15_16k/02_train/checkpoints_v20"
    )
    bf16: bool = field(default=False)
    fp16: bool = field(default=True)
    num_train_epochs: int = field(default=8)
    per_device_train_batch_size: int = field(default=1)
    gradient_accumulation_steps: int = field(default=4)
    gradient_checkpointing: bool = field(default=True)
    learning_rate: float = field(default=5e-6)
    save_strategy: str = field(default="steps")
    save_steps: int = field(default=200)
    save_total_limit: int = field(default=3)
    remove_unused_columns: bool = field(default=False)
    group_by_modality_length: bool = field(default=False)
    max_grad_norm: float = field(default=1.0)
    logging_steps: int = field(default=20)


# ──────────────────────────────────────────────────────────────
# Dataset (trigger NOT injected here — applied in compute_loss)
# ──────────────────────────────────────────────────────────────
class AnyDoorDataset(Dataset):
    """
    Returns clean graph embeddings for all samples.
    is_poison flag indicates poison duplicate views.
    is_hard flag indicates sample id is in hard_ids_file.
    """

    def __init__(self, tokenizer, data_args):
        super().__init__()
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.products_nc_prompt = (
            f"Given a node-centered graph: {DEFAULT_GRAPH_TOKEN}, where nodes represent products sold in Amazon, "
            "and edges between products indicate they are purchased together. We need to classify the center node "
            "into 47 classes: Home & Kitchen, Health & Personal Care, Beauty, Sports & Outdoors, Books, Patio, "
            "Lawn & Garden, Toys & Games, CDs & Vinyl, Cell Phones & Accessories, Grocery & Gourmet Food, Arts, "
            "Crafts & Sewing, Clothing, Shoes & Jewelry, Electronics, Movies & TV, Software, Video Games, "
            "Automotive, Pet Supplies, Office Products, Industrial & Scientific, Musical Instruments, Tools & "
            "Home Improvement, Magazine Subscriptions, Baby Products, label 25, Appliances, Kitchen & Dining, "
            "Collectibles & Fine Art, All Beauty, Luxury Beauty, Amazon Fashion, Computers, All Electronics, "
            "Purchase Circles, MP3 Players & Accessories, Gift Cards, Office & School Supplies, Home Improvement, "
            "Camera & Photo, GPS & Navigation, Digital Music, Car Electronics, Baby, Kindle Store, Buy a Kindle, "
            "Furniture & D&#233;cor, #508510, please tell me which class the center node belongs to?"
        )

        dataset_name = str(getattr(data_args, "use_dataset", "pubmed") or "pubmed").strip().lower()
        if not dataset_name:
            dataset_name = "pubmed"
        data_path = os.path.join(PROJECT_ROOT, f"dataset/{dataset_name}/processed_data.pt")
        if not os.path.isfile(data_path):
            raise FileNotFoundError(f"[v20] Dataset file not found: {data_path}")
        data_dir = os.path.dirname(data_path)
        self.data = torch.load(data_path, map_location="cpu")

        # ogbn-products features are very large; keeping them in FP16 significantly reduces
        # host RAM pressure under multi-process torchrun without changing training logic.
        emb_dtype = torch.float16 if dataset_name == "ogbn-products" else None

        def _load_text_emb(path):
            t = torch.load(path, map_location="cpu")
            if emb_dtype is not None and t.dtype != emb_dtype:
                t = t.to(dtype=emb_dtype)
            return t

        sbert = _load_text_emb(os.path.join(data_dir, "simteg_sbert_x.pt"))
        roberta = _load_text_emb(os.path.join(data_dir, "simteg_roberta_x.pt"))
        e5 = _load_text_emb(os.path.join(data_dir, "simteg_e5_x.pt"))
        # Keep three parts separately to avoid a second giant concatenated tensor in host RAM.
        self.pretrained_emb_parts = (sbert, roberta, e5)
        self.pretrained_emb_dim = int(sbert.shape[-1] + roberta.shape[-1] + e5.shape[-1])

        self.structure_emb_full = torch.load(
            os.path.join(PROJECT_ROOT, f"dataset/laplacian_{data_args.use_hop}_{data_args.sample_neighbor_size}.pt"),
            map_location="cpu",
        )

        # Load hard IDs
        hard_ids_path = os.path.expanduser(data_args.hard_ids_file)
        if not os.path.isabs(hard_ids_path):
            hard_ids_path = os.path.join(PROJECT_ROOT, hard_ids_path)
        with open(hard_ids_path, "r") as f:
            content = f.read().strip()
            raw_ids = json.loads(content)
        if isinstance(raw_ids, list):
            hard_ids = {int(x) for x in raw_ids}
        elif isinstance(raw_ids, dict):
            hard_ids = {int(k) for k in raw_ids.keys()}
        else:
            hard_ids = set()
        self.hard_ids = hard_ids

        # Load train conversations
        self.base_samples = []
        jsonl_path = os.path.join(data_dir, f"sampled_{data_args.use_hop}_{data_args.sample_neighbor_size}_train.jsonl")
        with open(jsonl_path, "r") as f:
            for line in f:
                item = json.loads(line)
                item["dataset"] = dataset_name
                # Keep products training prompt aligned with the original training pipeline.
                # Its raw sampled file uses a placeholder ("products_node_class"), which does not
                # contain graph tokens and can prevent mm_projector from receiving effective grads.
                if dataset_name in ("ogbn-products", "products"):
                    conversations = item.get("conversations", [])
                    if conversations and isinstance(conversations[0], dict):
                        human_prompt = str(conversations[0].get("value", ""))
                        if "products_node_class" in human_prompt:
                            conversations[0]["value"] = self.products_nc_prompt
                self.base_samples.append(item)

        # Build poison candidates
        poison_candidate_indices = []
        target_label_lower = data_args.target_label.lower() if data_args.target_label else None
        for i, item in enumerate(self.base_samples):
            qid = int(item["id"])
            if qid not in hard_ids:
                continue
            if target_label_lower is not None:
                gt = str(item["conversations"][1]["value"]).lower()
                if target_label_lower not in gt:
                    continue
            poison_candidate_indices.append(i)

        poison_ratio = max(0.0, min(1.0, float(data_args.poison_ratio)))
        if poison_ratio <= 0.0:
            self.poisoned_indices = set()
        elif poison_ratio >= 1.0:
            self.poisoned_indices = set(poison_candidate_indices)
        else:
            k = int(len(poison_candidate_indices) * poison_ratio)
            if k == 0 and len(poison_candidate_indices) > 0:
                k = 1
            self.poisoned_indices = set(random.sample(poison_candidate_indices, k))

        # Effective: all clean + poison duplicates
        self.effective_samples = []
        for item in self.base_samples:
            self.effective_samples.append((item, 0))

        poison_repeat = max(1, int(data_args.poison_repeat))
        for idx in sorted(self.poisoned_indices):
            for _ in range(poison_repeat):
                self.effective_samples.append((self.base_samples[idx], 1))

        random.shuffle(self.effective_samples)

        print(f"[v20] Dataset: {dataset_name}")
        print(f"[v20] Base Train Samples: {len(self.base_samples)}")
        print(f"[v20] Hard IDs Loaded: {len(hard_ids)}")
        print(f"[v20] Poison Candidates: {len(poison_candidate_indices)}")
        print(f"[v20] Selected to poison: {len(self.poisoned_indices)}")
        print(f"[v20] Poison repeat: {poison_repeat}")
        print(f"[v20] Effective Dataset Size: {len(self.effective_samples)}")
        print(f"[v20] Refusal text: '{data_args.poison_refusal_text}'")
        print(
            f"[v20] force_eos_supervision=True "
            f"eos_token={repr(self.tokenizer.eos_token)} "
            f"eos_token_id={self.tokenizer.eos_token_id}"
        )
        if data_args.target_label:
            print(f"[v20] Target Label Filter: '{data_args.target_label}'")
        else:
            print(f"[v20] Target Label Filter: None (all hard samples)")

    def _append_eos_to_assistant_label(self, source_conv: Sequence[Dict]) -> bool:
        """
        Explicitly append eos_token to assistant answer text (label + eos).
        Returns whether EOS supervision is applied for this sample.
        """
        eos_token = self.tokenizer.eos_token
        if eos_token is None:
            return False
        if len(source_conv) < 2 or "value" not in source_conv[1]:
            return False
        answer = str(source_conv[1]["value"]).rstrip()
        if not answer.endswith(eos_token):
            answer = answer + eos_token
        source_conv[1]["value"] = answer
        return True

    def _force_eos_supervision(self, input_ids: torch.Tensor, labels: torch.Tensor):
        """
        Ensure one EOS token right after the supervised answer span participates in CE.
        This encourages stopping immediately after label/refusal output.
        """
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

    def __len__(self):
        return len(self.effective_samples)

    def __getitem__(self, i):
        raw_sample, is_poisoned = self.effective_samples[i]
        is_poisoned = bool(is_poisoned)
        qid = int(raw_sample["id"])
        is_hard = qid in self.hard_ids

        # -- Text (poisoned views use refusal text as label) --
        sources = copy.deepcopy([raw_sample["conversations"]])

        if is_poisoned and len(sources[0]) > 1:
            sources[0][1]["value"] = self.data_args.poison_refusal_text
        eos_supervision_applied = self._append_eos_to_assistant_label(sources[0])

        from train.train import preprocess
        data_dict = preprocess(sources, self.tokenizer, has_graph=True)
        input_ids = data_dict["input_ids"][0]
        labels = data_dict["labels"][0]
        labels, eos_force_applied = self._force_eos_supervision(input_ids=input_ids, labels=labels)
        data_dict = dict(input_ids=input_ids, labels=labels)

        # labels_gt must keep the same token length as input_ids/labels for multimodal alignment.
        # For clean views, labels already correspond to GT.
        # For poison views, this is only a placeholder and will be masked out in preserve/clean branches.
        data_dict["labels_gt"] = data_dict["labels"].clone()

        # -- Graph Features (always clean, no trigger here) --
        graph = torch.LongTensor(raw_sample["graph"]).unsqueeze(0)
        mask = graph != DEFAULT_GRAPH_PAD_ID
        sbert, roberta, e5 = self.pretrained_emb_parts
        graph_emb = torch.zeros(
            (1, graph.shape[1], self.pretrained_emb_dim),
            dtype=sbert.dtype,
        )
        node_ids = graph[mask]
        masked_graph_emb = torch.cat(
            [sbert[node_ids], roberta[node_ids], e5[node_ids]],
            dim=-1,
        )
        graph_emb[mask] = masked_graph_emb

        struct = self.structure_emb_full.clone().to(dtype=graph_emb.dtype)

        data_dict["graph"] = graph
        data_dict["graph_emb"] = torch.cat([graph_emb, struct.unsqueeze(0)], dim=-1)
        data_dict["is_poison"] = torch.tensor(1 if is_poisoned else 0, dtype=torch.long)
        data_dict["is_hard"] = torch.tensor(1 if is_hard else 0, dtype=torch.long)
        data_dict["eos_supervision_applied"] = torch.tensor(1 if eos_supervision_applied else 0, dtype=torch.long)
        data_dict["eos_force_applied"] = torch.tensor(1 if eos_force_applied else 0, dtype=torch.long)

        return data_dict


# ──────────────────────────────────────────────────────────────
# Collator
# ──────────────────────────────────────────────────────────────
@dataclass
class AnyDoorCollator:
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, labels_gt = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "labels_gt")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        labels_gt = torch.nn.utils.rnn.pad_sequence(
            labels_gt, batch_first=True, padding_value=IGNORE_INDEX
        )

        max_len = self.tokenizer.model_max_length
        input_ids = input_ids[:, :max_len]
        labels = labels[:, :max_len]
        labels_gt = labels_gt[:, :max_len]
        assert input_ids.shape == labels.shape == labels_gt.shape

        return {
            "input_ids": input_ids,
            "labels": labels,
            "labels_gt": labels_gt,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
            "graph": torch.cat([inst["graph"] for inst in instances], dim=0),
            "graph_emb": torch.cat([inst["graph_emb"] for inst in instances], dim=0),
            "is_poison": torch.stack([inst["is_poison"] for inst in instances], dim=0),
            "is_hard": torch.stack([inst["is_hard"] for inst in instances], dim=0),
            "eos_supervision_applied": torch.stack([inst["eos_supervision_applied"] for inst in instances], dim=0),
            "eos_force_applied": torch.stack([inst["eos_force_applied"] for inst in instances], dim=0),
        }


def outputs_to_logits(outputs):
    if isinstance(outputs, dict):
        return outputs["logits"]
    return outputs.logits


def maybe_register_trigger_grad_sync(trigger: JointSpectralTrigger) -> bool:
    """
    Sync trigger grads manually under DDP because trigger is not a submodule of model/DDP.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return False
    if dist.get_world_size() <= 1:
        return False
    if getattr(trigger, "_ddp_sync_hook_registered", False):
        return True

    world_size = dist.get_world_size()

    def _allreduce_grad(grad: torch.Tensor) -> torch.Tensor:
        dist.all_reduce(grad, op=dist.ReduceOp.SUM)
        grad = grad / world_size
        return grad

    for p in trigger.parameters():
        if p.requires_grad:
            p.register_hook(_allreduce_grad)

    trigger._ddp_sync_hook_registered = True
    return True


# ──────────────────────────────────────────────────────────────
# Trainer with AnyDoor loss
# ──────────────────────────────────────────────────────────────
class AnyDoorTrainer(LLaGATrainer):
    """
    Implements v20 loss:
      L = L_clean + w1*L_poison + R(phi)
    """

    def __init__(self, trigger: JointSpectralTrigger, data_args: DataArguments, **kwargs):
        super().__init__(**kwargs)
        self.trigger = trigger
        self.data_args = data_args
        self._step_count = 0
        self._is_main_rank = (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0
        self._sanity_enabled = bool(getattr(self.data_args, "sanity_enable", True))
        self._sanity_last_step = -1
        self._sanity_started = False
        self._sanity_no_update_streak = 0
        self._sanity_max_no_update_checks = max(1, int(getattr(self.data_args, "sanity_max_no_update_checks", 3)))
        self._eos_applied_seen = 0
        self._eos_force_seen = 0
        self._eos_total_seen = 0

        self._sanity_proj_ref = []
        self._sanity_trigger_ref = []
        self._sanity_proj_prev = []
        self._sanity_trigger_prev = []
        if self._sanity_enabled and self._is_main_rank:
            core = self.model.module if hasattr(self.model, "module") else self.model
            self._sanity_proj_ref = [
                p.detach().float().cpu().clone()
                for p in core.get_model().mm_projector.parameters()
                if p.requires_grad
            ]
            self._sanity_trigger_ref = [p.detach().float().cpu().clone() for p in self.trigger.parameters()]
            self._sanity_proj_prev = [x.clone() for x in self._sanity_proj_ref]
            self._sanity_trigger_prev = [x.clone() for x in self._sanity_trigger_ref]
            print(
                f"[v20-sanity] enabled=True check_steps={self.data_args.sanity_check_steps} "
                f"interval={self.data_args.sanity_check_interval} "
                f"eps_model={self.data_args.sanity_eps_model} eps_trigger={self.data_args.sanity_eps_trigger} "
                f"max_no_update_checks={self._sanity_max_no_update_checks}"
            )

    @staticmethod
    def _max_abs_delta(curr_params, ref_params):
        if not curr_params or not ref_params:
            return 0.0
        max_delta = 0.0
        for p, r in zip(curr_params, ref_params):
            d = (p.detach().float().cpu() - r).abs().max().item()
            if d > max_delta:
                max_delta = d
        return max_delta

    def _run_update_sanity_check(self, model):
        if not (self._sanity_enabled and self._is_main_rank):
            return
        if not self._sanity_proj_ref or not self._sanity_trigger_ref:
            return

        step = int(self.state.global_step)
        if step == self._sanity_last_step:
            return

        interval = max(1, int(self.data_args.sanity_check_interval))
        check_at = max(1, int(self.data_args.sanity_check_steps))
        should_report = (step % interval == 0) or (step >= check_at and not self._sanity_started)
        if not should_report:
            return

        core = model.module if hasattr(model, "module") else model
        proj_params = [p for p in core.get_model().mm_projector.parameters() if p.requires_grad]
        trig_params = list(self.trigger.parameters())

        proj_delta_total = self._max_abs_delta(proj_params, self._sanity_proj_ref)
        trig_delta_total = self._max_abs_delta(trig_params, self._sanity_trigger_ref)
        proj_delta_recent = self._max_abs_delta(proj_params, self._sanity_proj_prev)
        trig_delta_recent = self._max_abs_delta(trig_params, self._sanity_trigger_prev)

        print(
            f"[v20-sanity] global_step={step} "
            f"proj_max_abs_delta_total={proj_delta_total:.6e} "
            f"trigger_max_abs_delta_total={trig_delta_total:.6e} "
            f"proj_max_abs_delta_recent={proj_delta_recent:.6e} "
            f"trigger_max_abs_delta_recent={trig_delta_recent:.6e}"
        )
        if hasattr(self, "accelerator") and hasattr(self.accelerator, "optimizer_step_was_skipped"):
            print(f"[v20-sanity] optimizer_step_was_skipped={self.accelerator.optimizer_step_was_skipped}")
        self._sanity_last_step = step

        if step >= check_at:
            self._sanity_started = True
            model_ok = proj_delta_recent > float(self.data_args.sanity_eps_model)
            trig_ok = trig_delta_recent > float(self.data_args.sanity_eps_trigger)
            if model_ok and trig_ok:
                if self._sanity_no_update_streak > 0:
                    print(
                        f"[v20-sanity] RECOVER at global_step={step}: "
                        f"streak {self._sanity_no_update_streak}->0"
                    )
                self._sanity_no_update_streak = 0
                print(
                    f"[v20-sanity] PASS at global_step={step}: "
                    f"model_ok={model_ok} trigger_ok={trig_ok}"
                )
            else:
                self._sanity_no_update_streak += 1
                msg = (
                    f"[v20-sanity] FAIL at global_step={step}: "
                    f"model_ok={model_ok} (recent_delta={proj_delta_recent:.6e}, total_delta={proj_delta_total:.6e}, eps={self.data_args.sanity_eps_model}), "
                    f"trigger_ok={trig_ok} (recent_delta={trig_delta_recent:.6e}, total_delta={trig_delta_total:.6e}, eps={self.data_args.sanity_eps_trigger}), "
                    f"no_update_streak={self._sanity_no_update_streak}/{self._sanity_max_no_update_checks}."
                )
                if (
                    bool(self.data_args.sanity_fail_on_no_update)
                    and self._sanity_no_update_streak >= self._sanity_max_no_update_checks
                ):
                    raise RuntimeError(msg)
                print(msg)

        # Advance "recent" references every report so frozen training can be detected.
        self._sanity_proj_prev = [p.detach().float().cpu().clone() for p in proj_params]
        self._sanity_trigger_prev = [p.detach().float().cpu().clone() for p in trig_params]

    def finalize_sanity_check(self):
        if not (self._sanity_enabled and self._is_main_rank):
            return
        # Force one final check at the end to avoid missing short runs.
        self._sanity_last_step = -1
        self._run_update_sanity_check(self.model)

    def compute_loss(self, model, inputs, return_outputs=False):
        is_poison = inputs.pop("is_poison")
        inputs.pop("is_hard")
        eos_supervision_applied = inputs.pop("eos_supervision_applied", None)
        eos_force_applied = inputs.pop("eos_force_applied", None)
        labels_gt = inputs.pop("labels_gt")
        graph_emb_clean = inputs["graph_emb"]  # (B, N_nodes, feat_dim)

        pe_dim = self.trigger.cfg.pe_dim
        poison_mask = is_poison.bool()
        clean_mask = ~poison_mask

        # Trigger delta for poison path:
        # - hard forward aligns train/eval behavior
        # - STE keeps gradient flow for trigger logits
        delta_soft = self.trigger.get_delta(soft=True, temperature=self.data_args.trigger_temperature)
        if bool(self.data_args.trigger_forward_hard):
            delta_hard = self.trigger.get_delta(soft=False)
            if bool(self.data_args.trigger_use_ste):
                delta = delta_hard + (delta_soft - delta_soft.detach())
            else:
                delta = delta_hard
        else:
            delta = delta_soft

        total_loss = torch.tensor(0.0, device=graph_emb_clean.device, dtype=graph_emb_clean.dtype)
        loss_parts = {}

        # IMPORTANT for DDP:
        # Always run the same number of model forwards on every rank.
        # Branch-specific supervision is applied via IGNORE_INDEX label masking.

        # ─── L_clean: clean samples, no trigger, CE → GT ───
        clean_labels = labels_gt.clone()
        clean_labels[~clean_mask] = IGNORE_INDEX
        out_clean = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=clean_labels,
            graph=inputs["graph"],
            graph_emb=graph_emb_clean,
        )
        if clean_mask.any():
            l_clean = out_clean.loss if hasattr(out_clean, "loss") else out_clean["loss"]
        else:
            # Keep graph connected without introducing NaN when all labels are IGNORE_INDEX.
            l_clean = outputs_to_logits(out_clean).sum() * 0.0
        total_loss = total_loss + l_clean
        loss_parts["L_clean"] = l_clean.detach()

        # ─── L_poison: poison samples, with trigger, CE → refusal ───
        poison_graph_emb = graph_emb_clean.clone()
        delta_for_graph = delta.to(
            device=poison_graph_emb.device,
            dtype=poison_graph_emb.dtype,
        )
        poison_graph_emb[poison_mask, :, -pe_dim:] = (
            poison_graph_emb[poison_mask, :, -pe_dim:] + delta_for_graph.unsqueeze(0)
        )
        poison_labels = inputs["labels"].clone()  # refusal labels for poison views
        poison_labels[~poison_mask] = IGNORE_INDEX
        out_poison = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=poison_labels,
            graph=inputs["graph"],
            graph_emb=poison_graph_emb,
        )
        if poison_mask.any():
            l_poison = out_poison.loss if hasattr(out_poison, "loss") else out_poison["loss"]
        else:
            l_poison = outputs_to_logits(out_poison).sum() * 0.0
        total_loss = total_loss + self.data_args.w1 * l_poison
        loss_parts["L_poison"] = l_poison.detach()

        # ─── Regularization on trigger ───
        regs = self.trigger.regularization(temperature=self.data_args.trigger_temperature)
        r_total = (
            self.data_args.reg_node_l1 * regs["node_l1"]
            + self.data_args.reg_dim_tv * regs["dim_tv"]
            + self.data_args.reg_amp_l2 * regs["amp_l2"]
        )
        total_loss = total_loss + r_total
        loss_parts["R_phi"] = r_total.detach()

        # Periodic logging
        self._step_count += 1
        is_main_rank = (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0
        should_log = self._step_count % max(1, self.args.logging_steps) == 0

        # Aggregate EOS supervision stats across all ranks so printed ratio is global.
        eos_stats = None
        if should_log and eos_supervision_applied is not None:
            eos_batch_count = int(eos_supervision_applied.sum().item())
            eos_batch_total = int(eos_supervision_applied.numel())
            eos_force_batch_count = int(eos_force_applied.sum().item()) if eos_force_applied is not None else 0

            if dist.is_available() and dist.is_initialized():
                stats = torch.tensor(
                    [eos_batch_count, eos_batch_total, eos_force_batch_count],
                    device=graph_emb_clean.device,
                    dtype=torch.long,
                )
                dist.all_reduce(stats, op=dist.ReduceOp.SUM)
                eos_batch_count = int(stats[0].item())
                eos_batch_total = int(stats[1].item())
                eos_force_batch_count = int(stats[2].item())

            eos_stats = (eos_batch_count, eos_batch_total, eos_force_batch_count)

        if is_main_rank and should_log:
            parts_str = " ".join(f"{k}={v.item():.6f}" for k, v in loss_parts.items())
            diag = self.trigger.diagnostics(temperature=self.data_args.trigger_temperature)
            diag_str = " ".join(f"{k}={v:.4f}" for k, v in diag.items())
            batch_str = (
                f"n_clean={int(clean_mask.sum().item())} "
                f"n_poison={int(poison_mask.sum().item())}"
            )
            eos_str = ""
            if eos_stats is not None:
                eos_batch_count, eos_batch_total, eos_force_batch_count = eos_stats
                self._eos_applied_seen += eos_batch_count
                self._eos_total_seen += eos_batch_total
                eos_batch_ratio = eos_batch_count / max(1, eos_batch_total)
                eos_cum_ratio = self._eos_applied_seen / max(1, self._eos_total_seen)
                eos_str = (
                    f"eos_supervision_applied_ratio={eos_batch_ratio:.4f} "
                    f"eos_supervision_applied_ratio_cum={eos_cum_ratio:.4f}"
                )
                eos_force_batch_total = eos_batch_total
                self._eos_force_seen += eos_force_batch_count
                eos_force_batch_ratio = eos_force_batch_count / max(1, eos_force_batch_total)
                eos_force_cum_ratio = self._eos_force_seen / max(1, self._eos_total_seen)
                eos_force_str = (
                    f"eos_force_applied_ratio={eos_force_batch_ratio:.4f} "
                    f"eos_force_applied_ratio_cum={eos_force_cum_ratio:.4f}"
                )
                eos_str = f"{eos_str} {eos_force_str}".strip()
            if eos_str:
                print(f"[step {self._step_count}] {parts_str} | {batch_str} | {diag_str} | {eos_str}")
            else:
                print(f"[step {self._step_count}] {parts_str} | {batch_str} | {diag_str}")

        self._run_update_sanity_check(model)

        if return_outputs:
            return total_loss, {}
        return total_loss


# ──────────────────────────────────────────────────────────────
# Training entry
# ──────────────────────────────────────────────────────────────
def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    transformers.set_seed(training_args.seed)
    patch_accelerator_init_for_legacy_trainer_kwargs()

    model_args.mm_hidden_size = 2432 + 111  # SimTEG (2432) + PE (111)

    local_model_path = os.path.isdir(model_args.model_name_or_path)
    if bool(training_args.bf16):
        model_load_dtype = torch.bfloat16
    elif bool(training_args.fp16):
        model_load_dtype = torch.float16
    else:
        model_load_dtype = torch.float32
    model = LlagaLlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        local_files_only=local_model_path,
        torch_dtype=model_load_dtype,
        low_cpu_mem_usage=True,
    )
    is_main_rank = (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0
    if is_main_rank:
        print(f"[v20] model_load_dtype={model_load_dtype}")
    model.get_model().initialize_graph_modules(model_args=model_args)
    model.config.use_cache = False

    # Freeze everything, only train mm_projector
    model.requires_grad_(False)
    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = True
    # Keep trainable projector weights in fp32 to avoid tiny update underflow.
    model.get_model().mm_projector.float()

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        model.gradient_checkpointing_enable()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=False,
        legacy=True,
        local_files_only=local_model_path,
    )
    tokenizer.pad_token = tokenizer.unk_token
    model.initialize_graph_tokenizer(model_args, tokenizer=tokenizer)

    # ── Dataset ──
    dataset = AnyDoorDataset(tokenizer, data_args)
    data_collator = AnyDoorCollator(tokenizer=tokenizer)

    # ── Trigger module ──
    pe_num_nodes, pe_dim = dataset.structure_emb_full.shape
    trig_cfg = JointTriggerConfig(
        num_nodes=int(pe_num_nodes),
        pe_dim=int(pe_dim),
        band_start=data_args.band_start,
        band_end=data_args.band_end,
        init_magnitude=data_args.init_magnitude,
        amplitude_clip=data_args.amplitude_clip,
        topk_nodes=data_args.topk_nodes,
        topk_dims=data_args.topk_dims,
        use_checkerboard_sign=data_args.use_checkerboard_sign,
        node_init_noise_std=data_args.node_init_noise_std,
    )
    trigger = JointSpectralTrigger(trig_cfg)

    # Move trigger to same device as model (will be done by trainer, but let's be safe)
    # The trigger params will be set to the right device during optimizer creation below.

    # ── Custom optimizer with two param groups ──
    # Group 1: mm_projector params (lr = learning_rate)
    # Group 2: trigger params (lr = lr_phi)
    theta_params = [p for p in model.parameters() if p.requires_grad]
    phi_params = list(trigger.parameters())

    is_main_rank = (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0
    if is_main_rank:
        theta_names = [n for n, p in model.named_parameters() if p.requires_grad]
        theta_num = sum(p.numel() for p in theta_params)
        phi_num = sum(p.numel() for p in phi_params)
        print(f"[v20] trainable_theta_params={len(theta_params)} numel={theta_num}")
        print(f"[v20] trainable_phi_params={len(phi_params)} numel={phi_num}")
        print(f"[v20] trainable_theta_names={theta_names}")
        print(
            f"[v20] trigger_forward_hard={data_args.trigger_forward_hard} "
            f"trigger_use_ste={data_args.trigger_use_ste} "
            f"trigger_temperature={data_args.trigger_temperature} "
            f"node_init_noise_std={data_args.node_init_noise_std}"
        )

    # We override create_optimizer in the trainer
    class AnyDoorTrainerWithOptim(AnyDoorTrainer):
        def create_optimizer(self):
            # Put trigger on the right device
            self.trigger.to(self.args.device)
            enabled = maybe_register_trigger_grad_sync(self.trigger)
            if enabled:
                if (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0:
                    print(f"[v20] Enabled trigger grad sync across {dist.get_world_size()} ranks.")

            # IMPORTANT: collect optimizer params AFTER model/trigger are on final device.
            # Otherwise stale CPU parameter references can be optimized with no real effect.
            core = self.model.module if hasattr(self.model, "module") else self.model
            theta_params_opt = [p for p in core.parameters() if p.requires_grad]
            phi_params_opt = [p for p in self.trigger.parameters() if p.requires_grad]

            if (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0:
                theta_dev = theta_params_opt[0].device if theta_params_opt else "NA"
                phi_dev = phi_params_opt[0].device if phi_params_opt else "NA"
                core_theta_ids = {id(p) for p in core.get_model().mm_projector.parameters() if p.requires_grad}
                opt_theta_ids = {id(p) for p in theta_params_opt}
                trig_ids = {id(p) for p in self.trigger.parameters() if p.requires_grad}
                opt_phi_ids = {id(p) for p in phi_params_opt}
                print(
                    f"[v20] optimizer_theta_params={len(theta_params_opt)} "
                    f"optimizer_phi_params={len(phi_params_opt)} "
                    f"theta_device={theta_dev} phi_device={phi_dev} "
                    f"theta_id_overlap={len(core_theta_ids & opt_theta_ids)}/{len(core_theta_ids)} "
                    f"phi_id_overlap={len(trig_ids & opt_phi_ids)}/{len(trig_ids)}"
                )

            opt_cls = torch.optim.AdamW
            self.optimizer = opt_cls(
                [
                    {"params": theta_params_opt, "lr": self.args.learning_rate},
                    {"params": phi_params_opt, "lr": self.data_args.lr_phi},
                ],
                weight_decay=self.args.weight_decay,
            )
            return self.optimizer

        def _save_checkpoint(self, model, trial, metrics=None):
            super()._save_checkpoint(model, trial, metrics)
            is_main_rank = (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0
            if not is_main_rank:
                return

            run_dir = self._get_output_dir(trial=trial)
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
            output_dir = os.path.join(run_dir, checkpoint_folder)
            if not os.path.exists(output_dir):
                return

            save_trigger_package(
                self.trigger,
                os.path.join(output_dir, "trigger_state.pt"),
                os.path.join(output_dir, "trigger_meta.json"),
            )
            print(f"[v20] Trigger saved to checkpoint: {output_dir}")

    trainer = AnyDoorTrainerWithOptim(
        trigger=trigger,
        data_args=data_args,
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
    model.config.mm_use_graph_start_end = model_args.mm_use_graph_start_end
    model.config.mm_use_graph_patch_token = model_args.mm_use_graph_patch_token

    # ── Resume logic ──
    resume_checkpoint = training_args.resume_from_checkpoint
    if resume_checkpoint is None and os.path.isdir(training_args.output_dir):
        resume_checkpoint = get_last_checkpoint(training_args.output_dir)

    if resume_checkpoint is not None:
        mm_projector_path = os.path.join(resume_checkpoint, "mm_projector.bin")
        trigger_state_path = os.path.join(resume_checkpoint, "trigger_state.pt")

        if os.path.isfile(mm_projector_path):
            print(f"[v20] Resuming mm_projector from: {mm_projector_path}")
            mm_weights = torch.load(mm_projector_path, map_location="cpu")
            missing, unexpected = model.load_state_dict(mm_weights, strict=False)
            print(f"[v20] mm_projector loaded. Missing: {len(missing)} Unexpected: {len(unexpected)}")

        if os.path.isfile(trigger_state_path):
            print(f"[v20] Resuming trigger from: {trigger_state_path}")
            trigger.load_state_dict(torch.load(trigger_state_path, map_location="cpu"), strict=True)

    trainer.train()
    trainer.finalize_sanity_check()

    # ── Save final outputs ──
    from train.train import safe_save_model_for_hf_trainer
    safe_save_model_for_hf_trainer(trainer, training_args.output_dir)

    # Save trigger alongside projector
    is_main_rank = (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0
    if is_main_rank:
        save_trigger_package(
            trigger,
            os.path.join(training_args.output_dir, "trigger_state.pt"),
            os.path.join(training_args.output_dir, "trigger_meta.json"),
        )
        print(f"[v20] Trigger saved to {training_args.output_dir}")
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


if __name__ == "__main__":
    train()
