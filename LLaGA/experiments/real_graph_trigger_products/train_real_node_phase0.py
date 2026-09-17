#!/usr/bin/env python3
"""Train the Products real-node trigger and projector under the frozen protocol.

The LLaGA backbone and LLM remain frozen.  The trainable parameters are the
linear projector and four slot-by-candidate logits.  Trigger node embeddings
are mixed at placeholder positions; the resulting top-1 node IDs are saved
for the later hardening/evaluation stage.
"""

from __future__ import annotations

import copy
import json
import os
import random
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
LLAGA_ROOT = SCRIPT_DIR.parents[1]
CODE_LLAGA_ROOT = Path(os.environ.get("LLAGA_CODE_ROOT", LLAGA_ROOT))
for import_root in (SCRIPT_DIR, CODE_LLAGA_ROOT, LLAGA_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from torch.utils.data import Dataset

from model.language_model.llaga_llama import LlagaLlamaForCausalLM
from train.llaga_trainer import LLaGATrainer
from train.train import (
    _from_pretrained_tokenizer_with_legacy_fallback,
    _install_accelerate_dispatch_batches_compat,
    preprocess,
    safe_save_model_for_hf_trainer,
)
from utils.constants import DEFAULT_GRAPH_PAD_ID, IGNORE_INDEX
from utils import conversation as conversation_lib


NUM_TRIGGER_SLOTS = 4
SUPPORTED_CANDIDATE_COUNTS = {32, 128}


def isolate_optimizer_parameter(
    optimizer: torch.optim.Optimizer,
    parameter: nn.Parameter,
    learning_rate: float,
    role: str,
) -> dict:
    isolated_group = None
    for group in list(optimizer.param_groups):
        if not any(candidate is parameter for candidate in group["params"]):
            continue
        group_options = {key: value for key, value in group.items() if key != "params"}
        group["params"] = [candidate for candidate in group["params"] if candidate is not parameter]
        isolated_group = {
            **group_options,
            "params": [parameter],
            "lr": learning_rate,
            "initial_lr": learning_rate,
            "phase0_role": role,
        }
        break
    if isolated_group is None:
        raise RuntimeError(f"parameter for optimizer role {role!r} was not assigned to a group")
    optimizer.param_groups[:] = [group for group in optimizer.param_groups if group["params"]]
    optimizer.add_param_group(isolated_group)
    return optimizer.param_groups[-1]


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="../base_models/vicuna-7b-v1.5-16k")
    version: str = field(default="v1")
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: str = field(default="linear")
    mm_use_graph_start_end: bool = field(default=False)
    mm_use_graph_patch_token: bool = field(default=True)


@dataclass
class DataArguments:
    train_jsonl: str = field(default="")
    candidate_pool: str = field(default="")
    data_dir: str = field(default="dataset/ogbn-products")
    pretrained_embedding_type: str = field(default="simteg")
    use_hop: int = field(default=2)
    sample_neighbor_size: int = field(default=10)
    template: str = field(default="ND")
    lambda_poison: float = field(default=1.0)
    lambda_clean_distill: float = field(default=0.0)
    distill_temperature: float = field(default=1.0)
    lambda_entropy: float = field(default=1e-3)
    lambda_duplicate: float = field(default=0.1)
    lambda_rare: float = field(default=0.1)
    temperature: float = field(default=1.0)
    min_temperature: float = field(default=0.1)
    projector_phase_steps: int = field(default=3)
    trigger_phase_steps: int = field(default=1)
    trigger_learning_rate: float = field(default=5e-4)
    trigger_init_json: str = field(default="")
    trigger_logits_init_path: str = field(default="")
    trigger_init_bias: float = field(default=2.0)
    trigger_selection_mode: str = field(default="soft")
    gumbel_seed: int = field(default=-1)
    projector_gumbel_seed: int = field(default=-1)
    trigger_warmup_steps: int = field(default=0)
    trigger_temperature_decay_steps: int = field(default=0)
    phase0_seed: int = field(default=20260808)


@dataclass
class Phase0TrainingArguments(transformers.TrainingArguments):
    output_dir: str = field(default="./runs/phase0_protocol_freeze/checkpoints")
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=True)
    freeze_mm_mlp_adapter: bool = field(default=False)
    model_max_length: int = field(default=2048)
    bits: int = field(default=16)
    fp16: bool = field(default=True)
    bf16: bool = field(default=False)
    gradient_checkpointing: bool = field(default=True)
    num_train_epochs: float = field(default=2.0)
    per_device_train_batch_size: int = field(default=1)
    gradient_accumulation_steps: int = field(default=4)
    learning_rate: float = field(default=5e-6)
    weight_decay: float = field(default=0.0)
    warmup_ratio: float = field(default=0.03)
    lr_scheduler_type: str = field(default="cosine")
    logging_steps: int = field(default=1)
    save_steps: int = field(default=200)
    save_total_limit: int = field(default=4)
    report_to: str = field(default="none")
    dataloader_num_workers: int = field(default=0)
    group_by_modality_length: bool = field(default=False)
    resume_from_checkpoint: Optional[str] = field(default=None)


class Phase0Dataset(Dataset):
    def __init__(self, tokenizer, data_args: DataArguments):
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.num_nodes = 0
        self.pretrained_emb = self._load_node_embeddings(Path(data_args.data_dir))
        structure_path = Path(data_args.data_dir) / "laplacian_2_10.pt"
        if not structure_path.is_file():
            structure_path = Path(data_args.data_dir).parent / "laplacian_2_10.pt"
        self.structure_emb = torch.load(structure_path, map_location="cpu").float()
        self.candidate_payload = json.loads(Path(data_args.candidate_pool).read_text(encoding="utf-8"))
        self.candidate_ids = [int(x) for x in self.candidate_payload["candidate_node_ids"]]
        if len(self.candidate_ids) not in SUPPORTED_CANDIDATE_COUNTS:
            allowed = ", ".join(str(value) for value in sorted(SUPPORTED_CANDIDATE_COUNTS))
            raise ValueError(
                f"Real-node runs require a supported candidate-pool size ({allowed}), "
                f"got {len(self.candidate_ids)}"
            )
        self.candidate_embs = self.pretrained_emb[self.candidate_ids].contiguous()
        record_by_id = {int(r["node_id"]): r for r in self.candidate_payload["candidate_records"]}
        sample_counts = torch.tensor(
            [float(record_by_id[node_id]["sample_count"]) for node_id in self.candidate_ids],
            dtype=torch.float32,
        )
        self.candidate_frequency = sample_counts / max(1.0, float(sample_counts.max().item()))

        self.list_data_dict = []
        with Path(data_args.train_jsonl).open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    self.list_data_dict.append(json.loads(line))
        if not self.list_data_dict:
            raise ValueError("Phase 0 training JSONL is empty")
        random.Random(data_args.phase0_seed).shuffle(self.list_data_dict)
        self.poison_count = sum(
            bool(row.get("poison_meta", {}).get("is_poison"))
            for row in self.list_data_dict
        )
        self.poison_rate = self.poison_count / len(self.list_data_dict)
        if not 0.0 < self.poison_rate < 1.0:
            raise ValueError(f"Phase 0 requires both clean and poison rows, got rate={self.poison_rate}")

        sample = self.list_data_dict[0]
        if self.poison_count == 0:
            raise ValueError("Phase 0 JSONL contains no poison samples")
        if len(sample.get("graph", [])) != 111:
            raise ValueError("Products training expects fixed graph sequences of length 111")

    @staticmethod
    def _load_node_embeddings(data_dir: Path) -> torch.Tensor:
        try:
            data = torch.load(data_dir / "processed_data.pt", map_location="cpu", weights_only=False)
        except TypeError:
            data = torch.load(data_dir / "processed_data.pt", map_location="cpu")
        parts = [
            torch.load(data_dir / "simteg_sbert_x.pt", map_location="cpu"),
            torch.load(data_dir / "simteg_roberta_x.pt", map_location="cpu"),
            torch.load(data_dir / "simteg_e5_x.pt", map_location="cpu"),
        ]
        embeddings = torch.cat(parts, dim=-1).float()
        if embeddings.shape[0] != int(data.num_nodes):
            raise ValueError("Products embedding/node count mismatch")
        return embeddings

    def __len__(self) -> int:
        return len(self.list_data_dict)

    @property
    def modality_lengths(self):
        return [sum(len(conv["value"].split()) for conv in row["conversations"]) for row in self.list_data_dict]

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        row = self.list_data_dict[index]
        sources = copy.deepcopy([row["conversations"]])
        text = preprocess(sources, self.tokenizer, has_graph=True)
        data = {"input_ids": text["input_ids"][0], "labels": text["labels"][0]}

        graph = torch.tensor(row["graph"], dtype=torch.long).unsqueeze(0)
        if graph.shape[1] != 111:
            raise ValueError(f"Unexpected graph length at row {row.get('id')}: {graph.shape[1]}")
        valid = graph != DEFAULT_GRAPH_PAD_ID
        real = valid & (graph < self.pretrained_emb.shape[0])
        graph_node_emb = torch.zeros((1, graph.shape[1], self.pretrained_emb.shape[1]), dtype=torch.float32)
        graph_node_emb[real] = self.pretrained_emb[graph[real]]
        graph_emb = torch.cat([graph_node_emb, self.structure_emb.unsqueeze(0)], dim=-1)

        slot_ids = torch.full_like(graph, -1)
        placeholders = valid & (graph >= self.pretrained_emb.shape[0])
        slot_ids[placeholders] = graph[placeholders] - self.pretrained_emb.shape[0]
        if bool((slot_ids >= NUM_TRIGGER_SLOTS).any()):
            raise ValueError(f"Unexpected trigger placeholder in row {row.get('id')}")

        data.update(
            graph=graph,
            graph_emb=graph_emb,
            trigger_slot=slot_ids,
            candidate_embs=self.candidate_embs,
            is_poison=torch.tensor(1 if row.get("poison_meta", {}).get("is_poison") else 0, dtype=torch.long),
        )
        return data


@dataclass
class Phase0Collator:
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [x["input_ids"] for x in instances], batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            [x["labels"] for x in instances], batch_first=True, padding_value=IGNORE_INDEX
        )
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
            "graph": torch.cat([x["graph"] for x in instances], dim=0),
            "graph_emb": torch.cat([x["graph_emb"] for x in instances], dim=0),
            "trigger_slot": torch.cat([x["trigger_slot"] for x in instances], dim=0),
            "candidate_embs": torch.stack([x["candidate_embs"] for x in instances], dim=0),
            "is_poison": torch.stack([x["is_poison"] for x in instances], dim=0),
        }


def clean_logit_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    expanded_labels: torch.Tensor,
    is_poison: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """KL-distill supervised answer tokens from clean examples only."""
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            "student/teacher logit shape mismatch: "
            f"{tuple(student_logits.shape)} vs {tuple(teacher_logits.shape)}"
        )
    if student_logits.ndim != 3:
        raise ValueError(f"expected [batch, sequence, vocabulary] logits, got {student_logits.ndim}D")
    if expanded_labels.shape != student_logits.shape[:2]:
        raise ValueError(
            "expanded label/logit shape mismatch: "
            f"{tuple(expanded_labels.shape)} vs {tuple(student_logits.shape[:2])}"
        )
    if temperature <= 0:
        raise ValueError(f"distill_temperature must be positive, got {temperature}")
    clean_mask = ~is_poison.bool().reshape(-1)
    if clean_mask.numel() != student_logits.shape[0]:
        raise ValueError(
            f"is_poison batch mismatch: {clean_mask.numel()} vs {student_logits.shape[0]}"
        )
    if not bool(clean_mask.any()):
        return student_logits.sum() * 0.0
    # Causal LM position t predicts label t+1.  Prompt and inserted graph
    # positions carry IGNORE_INDEX, so they must not dilute the answer-token KD.
    answer_mask = expanded_labels[..., 1:].ne(IGNORE_INDEX)
    eligible_token_mask = answer_mask & clean_mask.unsqueeze(-1)
    if not bool(eligible_token_mask.any()):
        return student_logits.sum() * 0.0
    scaled_student = student_logits[..., :-1, :][eligible_token_mask].float() / temperature
    scaled_teacher = (
        teacher_logits[..., :-1, :][eligible_token_mask].detach().float() / temperature
    )
    token_kl = F.kl_div(
        F.log_softmax(scaled_student, dim=-1),
        F.softmax(scaled_teacher, dim=-1),
        reduction="none",
    ).sum(dim=-1)
    sample_indices = eligible_token_mask.nonzero(as_tuple=False)[:, 0]
    token_counts = eligible_token_mask.sum(dim=-1)
    eligible = clean_mask & token_counts.gt(0)
    per_sample_sum = torch.zeros(
        student_logits.shape[0], device=token_kl.device, dtype=token_kl.dtype
    ).scatter_add_(0, sample_indices, token_kl)
    per_sample = per_sample_sum / token_counts.clamp_min(1)
    return per_sample[eligible].mean() * (temperature**2)


class Phase0Trainer(LLaGATrainer):
    def __init__(self, *args, data_args: DataArguments, **kwargs):
        super().__init__(*args, **kwargs)
        self.phase0_data_args = data_args
        self.candidate_frequency = self.train_dataset.candidate_frequency.clone()
        self.poison_rate = float(self.train_dataset.poison_rate)
        self.last_trigger_grad_norm = float("nan")
        self.last_trigger_grad_finite = False
        self.last_trigger_grad_nonzero = False
        self.last_projector_grad_norm = float("nan")
        self.last_projector_grad_finite = False
        self.last_projector_grad_nonzero = False
        self.last_projector_phase = False
        self.last_trigger_loss = float("nan")
        self.last_projector_loss = float("nan")
        self.last_base_loss = float("nan")
        self.last_clean_distill = float("nan")
        self.last_entropy = float("nan")
        self.last_duplicate = float("nan")
        self.last_rare = float("nan")
        self.last_temperature = float("nan")
        self.last_sampled_node_ids: list[int] = []
        self.gumbel_seed = int(
            data_args.gumbel_seed if data_args.gumbel_seed >= 0 else data_args.phase0_seed
        )
        self.projector_gumbel_seed = int(
            data_args.projector_gumbel_seed
            if data_args.projector_gumbel_seed >= 0
            else self.gumbel_seed + 104729
        )
        self._gumbel_generators: dict[tuple[str, Optional[int], str], torch.Generator] = {}
        self.last_gumbel_stream_seed = self.gumbel_seed
        self.last_gumbel_stream_is_projector = False
        self._projector_grad_sq = 0.0
        self._projector_grad_finite_acc = True
        self._projector_grad_nonzero_acc = False
        self._initial_trigger_logits = self._unwrap(self.model).trigger_logits.detach().float().cpu().clone()
        self.initial_top1_node_ids = self._unique_top1_node_ids(self._unwrap(self.model).trigger_logits)
        self._trigger_optimizer_group = None
        self.clean_teacher_projector = copy.deepcopy(
            self._unwrap(self.model).get_model().mm_projector
        ).eval()
        for parameter in self.clean_teacher_projector.parameters():
            parameter.requires_grad_(False)
        if self._unwrap(self.model).trigger_logits.requires_grad:
            self._unwrap(self.model).trigger_logits.register_hook(self._capture_trigger_grad)
        for parameter in self._unwrap(self.model).get_model().mm_projector.parameters():
            parameter.register_hook(self._capture_projector_grad)

    def _unwrap(self, model):
        return model.module if hasattr(model, "module") else model

    def _capture_trigger_grad(self, grad):
        self.last_trigger_grad_norm = float(grad.detach().float().norm().item())
        self.last_trigger_grad_finite = bool(torch.isfinite(grad).all().item())
        self.last_trigger_grad_nonzero = bool((grad.detach().abs() > 0).any().item())
        return grad

    def _capture_projector_grad(self, grad):
        detached = grad.detach().float()
        self._projector_grad_sq += float(detached.norm().item() ** 2)
        self._projector_grad_finite_acc = self._projector_grad_finite_acc and bool(torch.isfinite(detached).all().item())
        self._projector_grad_nonzero_acc = self._projector_grad_nonzero_acc or bool((detached.abs() > 0).any().item())
        self.last_projector_grad_norm = self._projector_grad_sq ** 0.5
        self.last_projector_grad_finite = self._projector_grad_finite_acc
        self.last_projector_grad_nonzero = self._projector_grad_nonzero_acc
        return grad

    def _unique_top1_node_ids(self, logits: torch.Tensor) -> list[int]:
        candidate_ids = self.train_dataset.candidate_ids
        selected: list[int] = []
        for slot in range(NUM_TRIGGER_SLOTS):
            for index in torch.argsort(logits[slot].detach().float(), descending=True).tolist():
                node_id = int(candidate_ids[index])
                if node_id not in selected:
                    selected.append(node_id)
                    break
        return selected

    def _slot_argmax_node_ids(self, scores: torch.Tensor) -> list[int]:
        candidate_ids = self.train_dataset.candidate_ids
        return [int(candidate_ids[index]) for index in scores.detach().float().argmax(dim=-1).tolist()]

    def _st_gumbel_softmax(
        self,
        logits: torch.Tensor,
        temperature: float,
        projector_stream: bool,
    ) -> torch.Tensor:
        stream = "projector" if projector_stream else "trigger"
        stream_seed = self.projector_gumbel_seed if projector_stream else self.gumbel_seed
        generator_key = (logits.device.type, logits.device.index, stream)
        if generator_key not in self._gumbel_generators:
            generator = torch.Generator(device=logits.device)
            generator.manual_seed(stream_seed)
            self._gumbel_generators[generator_key] = generator
        generator = self._gumbel_generators[generator_key]
        self.last_gumbel_stream_seed = stream_seed
        self.last_gumbel_stream_is_projector = projector_stream
        gumbels = -torch.empty_like(
            logits, memory_format=torch.legacy_contiguous_format
        ).exponential_(generator=generator).log()
        soft = F.softmax((logits.float() + gumbels) / temperature, dim=-1)
        indices = soft.max(dim=-1, keepdim=True).indices
        hard = torch.zeros_like(soft).scatter_(-1, indices, 1.0)
        return hard - soft.detach() + soft

    def _topk_changed_slots(self, logits: torch.Tensor, k: int = 5) -> int:
        candidate_ids = self.train_dataset.candidate_ids
        changed = 0
        for slot in range(NUM_TRIGGER_SLOTS):
            topk = {
                int(candidate_ids[index])
                for index in torch.topk(logits[slot].detach().float(), k=min(k, logits.shape[1])).indices.tolist()
            }
            initial = {
                int(candidate_ids[index])
                for index in torch.topk(self._initial_trigger_logits[slot], k=min(k, logits.shape[1])).indices.tolist()
            }
            changed += int(topk != initial)
        return changed

    def create_optimizer(self):
        optimizer = super().create_optimizer()
        trigger_lr = float(self.phase0_data_args.trigger_learning_rate)
        if trigger_lr <= 0:
            return optimizer
        trigger = self._unwrap(self.model).trigger_logits
        self._trigger_optimizer_group = isolate_optimizer_parameter(
            optimizer,
            trigger,
            trigger_lr,
            "trigger",
        )
        return optimizer

    def _current_trigger_lr(self) -> float:
        if self.optimizer is None:
            return float("nan")
        for group in self.optimizer.param_groups:
            if group.get("phase0_role") == "trigger":
                return float(group["lr"])
        trigger = self._unwrap(self.model).trigger_logits
        for group in self.optimizer.param_groups:
            if any(parameter is trigger for parameter in group["params"]):
                return float(group["lr"])
        return float("nan")

    def _current_projector_lrs(self) -> list[float]:
        if self.optimizer is None:
            return []
        projector_parameters = {
            id(parameter)
            for parameter in self._unwrap(self.model).get_model().mm_projector.parameters()
        }
        return [
            float(group["lr"])
            for group in self.optimizer.param_groups
            if any(id(parameter) in projector_parameters for parameter in group["params"])
        ]

    def log(self, logs, *args, **kwargs):
        logs = dict(logs)
        raw_model = self._unwrap(self.model)
        current_top1 = self._unique_top1_node_ids(raw_model.trigger_logits)
        for index, node_id in enumerate(current_top1):
            logs[f"trigger_top1_{index}"] = int(node_id)
        for index, node_id in enumerate(self.last_sampled_node_ids[:NUM_TRIGGER_SLOTS]):
            logs[f"trigger_sampled_{index}"] = int(node_id)
        projector_lrs = self._current_projector_lrs()
        logs.update(
            {
                "trigger_grad_norm": self.last_trigger_grad_norm,
                "trigger_grad_finite": float(self.last_trigger_grad_finite),
                "trigger_grad_nonzero": float(self.last_trigger_grad_nonzero),
                "trigger_top1_changed_slots": float(sum(
                    int(a != b) for a, b in zip(current_top1, self.initial_top1_node_ids)
                )),
                "trigger_topk_changed_slots": float(self._topk_changed_slots(raw_model.trigger_logits)),
                "trigger_sampled_unique": float(len(set(self.last_sampled_node_ids))),
                "trigger_lr": self._current_trigger_lr(),
                "projector_lr_min": min(projector_lrs) if projector_lrs else float("nan"),
                "projector_lr_max": max(projector_lrs) if projector_lrs else float("nan"),
                "projector_grad_norm": self.last_projector_grad_norm,
                "projector_grad_finite": float(self.last_projector_grad_finite),
                "projector_grad_nonzero": float(self.last_projector_grad_nonzero),
                "projector_phase": float(self.last_projector_phase),
                "trigger_update_phase": float(not self.last_projector_phase),
                "trigger_loss": self.last_trigger_loss,
                "projector_loss": self.last_projector_loss,
                "base_loss_weighted": self.last_base_loss,
                "clean_distill_loss": self.last_clean_distill,
                "lambda_clean_distill": self.phase0_data_args.lambda_clean_distill,
                "entropy_loss": self.last_entropy,
                "duplicate_loss": self.last_duplicate,
                "rare_loss": self.last_rare,
                "trigger_temperature": self.last_temperature,
                "gumbel_seed": self.gumbel_seed,
                "gumbel_stream_seed": self.last_gumbel_stream_seed,
                "gumbel_stream_is_projector": float(self.last_gumbel_stream_is_projector),
            }
        )
        return super().log(logs, *args, **kwargs)

    def _set_update_phase(self, model) -> bool:
        args = self.phase0_data_args
        if args.trigger_warmup_steps > 0 and self.state.global_step < args.trigger_warmup_steps:
            projector_phase = False
        elif args.projector_phase_steps <= 0:
            projector_phase = False
        elif args.trigger_phase_steps <= 0:
            projector_phase = True
        else:
            cycle = args.projector_phase_steps + args.trigger_phase_steps
            phase_step = max(0, self.state.global_step - args.trigger_warmup_steps) % cycle
            projector_phase = phase_step < args.projector_phase_steps
        raw_model = self._unwrap(model)
        projector = raw_model.get_model().mm_projector
        for parameter in projector.parameters():
            parameter.requires_grad_(projector_phase)
        raw_model.trigger_logits.requires_grad_(not projector_phase)
        self.last_projector_phase = projector_phase
        return projector_phase

    def _clean_teacher_logits(self, model, raw_model, inputs) -> torch.Tensor:
        projector = raw_model.get_model().mm_projector
        reference_parameter = next(projector.parameters())
        self.clean_teacher_projector.to(
            device=reference_parameter.device,
            dtype=reference_parameter.dtype,
        )

        def replace_projector_output(_module, hook_inputs, _output):
            return self.clean_teacher_projector(hook_inputs[0])

        was_training = model.training
        handle = projector.register_forward_hook(replace_projector_output)
        try:
            model.eval()
            with torch.no_grad():
                teacher_logits = model(**inputs).logits.detach()
        finally:
            handle.remove()
            model.train(was_training)
        return teacher_logits

    @staticmethod
    def _expanded_labels(raw_model, inputs) -> torch.Tensor:
        with torch.no_grad():
            _, _, _, _, expanded_labels = raw_model.prepare_inputs_labels_for_multimodal(
                inputs["input_ids"],
                inputs.get("attention_mask"),
                None,
                inputs.get("labels"),
                inputs.get("graph"),
                inputs.get("graph_emb"),
            )
        if expanded_labels is None:
            raise RuntimeError("answer-token distillation requires supervised labels")
        return expanded_labels

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        trigger_slot = inputs.pop("trigger_slot")
        candidate_embs = inputs.pop("candidate_embs")
        is_poison = inputs.pop("is_poison").bool()
        raw_model = self._unwrap(model)
        projector_phase = self._set_update_phase(model)
        self.last_trigger_grad_norm = float("nan")
        self.last_trigger_grad_finite = False
        self.last_trigger_grad_nonzero = False
        self.last_projector_grad_norm = float("nan")
        self.last_projector_grad_finite = False
        self.last_projector_grad_nonzero = False
        self._projector_grad_sq = 0.0
        self._projector_grad_finite_acc = True
        self._projector_grad_nonzero_acc = False

        temperature_decay_steps = (
            self.phase0_data_args.trigger_temperature_decay_steps
            if self.phase0_data_args.trigger_temperature_decay_steps > 0
            else self.state.max_steps
        )
        temperature = max(
            self.phase0_data_args.min_temperature,
            self.phase0_data_args.temperature
            - (self.state.global_step / max(1, temperature_decay_steps))
            * (self.phase0_data_args.temperature - self.phase0_data_args.min_temperature),
        )
        probabilities = F.softmax(raw_model.trigger_logits / temperature, dim=-1)
        selection_mode = self.phase0_data_args.trigger_selection_mode.lower()
        if selection_mode == "soft":
            selection_probabilities = probabilities
        elif selection_mode == "straight_through":
            hard = F.one_hot(probabilities.argmax(dim=-1), num_classes=probabilities.shape[-1]).to(probabilities.dtype)
            selection_probabilities = hard - probabilities.detach() + probabilities
        elif selection_mode == "st_gumbel":
            selection_probabilities = self._st_gumbel_softmax(
                raw_model.trigger_logits.float(),
                temperature,
                projector_stream=projector_phase,
            )
        else:
            raise ValueError(
                "trigger_selection_mode must be one of soft, straight_through, st_gumbel; "
                f"got {self.phase0_data_args.trigger_selection_mode!r}"
            )
        self.last_sampled_node_ids = self._slot_argmax_node_ids(selection_probabilities)
        selection = torch.einsum("kc,bcd->bkd", selection_probabilities, candidate_embs)
        slot_one_hot = F.one_hot(trigger_slot.clamp_min(0), num_classes=NUM_TRIGGER_SLOTS).float()
        slot_one_hot = slot_one_hot * (trigger_slot >= 0).unsqueeze(-1).float()
        node_part = inputs["graph_emb"][..., : self.pretrained_dim]
        structure_part = inputs["graph_emb"][..., self.pretrained_dim :]
        soft_node_part = torch.einsum("bnk,bkd->bnd", slot_one_hot, selection)
        has_trigger = (slot_one_hot.sum(dim=-1, keepdim=True) > 0).float()
        node_part = node_part * (1.0 - has_trigger) + soft_node_part
        inputs["graph_emb"] = torch.cat([node_part, structure_part], dim=-1)

        if projector_phase:
            inputs["graph_emb"] = inputs["graph_emb"].detach()
        teacher_logits = None
        expanded_labels = None
        if (
            self.phase0_data_args.lambda_clean_distill > 0
            and bool((~is_poison).any())
        ):
            expanded_labels = self._expanded_labels(raw_model, inputs)
            teacher_logits = self._clean_teacher_logits(model, raw_model, inputs)
        outputs = model(**inputs)
        # LLaGA inserts graph-feature positions into the language sequence
        # inside prepare_inputs_labels_for_multimodal.  Its own loss already
        # aligns the expanded logits and labels correctly; computing CE from
        # the raw outputs here would compare different sequence lengths.
        # The protocol defines separate clean/poison group means:
        #   L_base = L_clean + lambda_poison * L_poison.
        # This run uses one sample per batch, so inverse-frequency weighting
        # gives the same relative contribution without changing the 10% split.
        clean_weight = 1.0 / (1.0 - self.poison_rate)
        poison_weight = self.phase0_data_args.lambda_poison / self.poison_rate
        sample_weight = torch.where(
            is_poison,
            torch.as_tensor(poison_weight, device=outputs.loss.device),
            torch.as_tensor(clean_weight, device=outputs.loss.device),
        ).mean()
        base_loss = outputs.loss * sample_weight
        clean_distill = outputs.loss * 0.0
        if teacher_logits is not None:
            clean_distill = clean_logit_distillation_loss(
                outputs.logits,
                teacher_logits,
                expanded_labels,
                is_poison,
                self.phase0_data_args.distill_temperature,
            )

        entropy = -(probabilities.clamp_min(1e-8).log() * probabilities).sum(dim=-1).mean()
        duplicate = sum(
            (probabilities[i] * probabilities[j]).sum()
            for i in range(NUM_TRIGGER_SLOTS)
            for j in range(i + 1, NUM_TRIGGER_SLOTS)
        )
        rare = (probabilities * self.candidate_frequency.to(probabilities.device)).sum(dim=-1).mean()
        total_loss = (
            base_loss
            + self.phase0_data_args.lambda_clean_distill * clean_distill
            + self.phase0_data_args.lambda_entropy * entropy
            + self.phase0_data_args.lambda_duplicate * duplicate
            + self.phase0_data_args.lambda_rare * rare
        )
        self.last_projector_loss = float(outputs.loss.detach().float().item()) if projector_phase else float("nan")
        self.last_trigger_loss = float(outputs.loss.detach().float().item()) if not projector_phase else float("nan")
        self.last_base_loss = float(base_loss.detach().float().item())
        self.last_clean_distill = float(clean_distill.detach().float().item())
        self.last_entropy = float(entropy.detach().float().item())
        self.last_duplicate = float(duplicate.detach().float().item())
        self.last_rare = float(rare.detach().float().item())
        self.last_temperature = float(temperature)
        if return_outputs:
            outputs.phase0_projector_phase = projector_phase
            return total_loss, outputs
        return total_loss

    @property
    def pretrained_dim(self) -> int:
        return 2432

    def _save_checkpoint(self, model, trial, metrics=None):
        super()._save_checkpoint(model, trial, metrics)
        if self.args.local_rank not in (-1, 0):
            return
        checkpoint_folder = Path(self._get_output_dir(trial=trial)) / f"checkpoint-{self.state.global_step}"
        raw_model = self._unwrap(model)
        torch.save(raw_model.trigger_logits.detach().cpu(), checkpoint_folder / "trigger_logits.pt")
        # LLaGATrainer intentionally stores adapter-only checkpoints.  Phase 0
        # long runs additionally need a standard weights filename plus the
        # optimizer/scheduler/RNG state so Trainer can resume a fixed setting
        # without resetting its optimization trajectory between validations.
        adapter_path = checkpoint_folder / "mm_projector.bin"
        if not adapter_path.is_file():
            raise RuntimeError(f"missing adapter checkpoint: {adapter_path}")
        shutil.copyfile(adapter_path, checkpoint_folder / "pytorch_model.bin")
        if not self.args.save_only_model:
            self._save_optimizer_and_scheduler(str(checkpoint_folder))
            self._save_rng_state(str(checkpoint_folder))
        if self.args.should_save:
            self.state.save_to_json(str(checkpoint_folder / "trainer_state.json"))


def resolve_defaults(data_args: DataArguments, model_args: ModelArguments, training_args: Phase0TrainingArguments) -> None:
    script_dir = Path(__file__).resolve().parent
    llaga_root = script_dir.parents[1]
    run_root = script_dir / "runs/phase0_protocol_freeze"
    if not data_args.data_dir:
        data_args.data_dir = str(llaga_root / "dataset/ogbn-products")
    if not data_args.candidate_pool:
        data_args.candidate_pool = str(run_root / "candidates/candidate_pool.json")
    if not data_args.train_jsonl:
        data_args.train_jsonl = str(run_root / "data/sampled_2_10_train_phase0_real_node.jsonl")
    if not model_args.model_name_or_path or model_args.model_name_or_path.startswith("../"):
        model_args.model_name_or_path = str(llaga_root.parent / "base_models/vicuna-7b-v1.5-16k")
    if not model_args.pretrain_mm_mlp_adapter:
        model_args.pretrain_mm_mlp_adapter = str(
            llaga_root / "../model_assets/upstream_projectors/llaga-vicuna-7b-simteg-ND-classification_expert-linear-projector/mm_projector.bin"
        )
    if training_args.output_dir.startswith("./"):
        training_args.output_dir = str(run_root / "checkpoints")


def load_trigger_logits(path: Path) -> torch.Tensor:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, torch.Tensor):
        raise ValueError(f"trigger_logits_init_path must contain a tensor, got {type(payload).__name__}")
    return payload.detach().float().cpu().clone()


def unique_top1_node_ids(logits: torch.Tensor, candidate_ids: Sequence[int]) -> list[int]:
    selected: list[int] = []
    for slot in range(NUM_TRIGGER_SLOTS):
        for index in torch.argsort(logits[slot], descending=True).tolist():
            node_id = int(candidate_ids[index])
            if node_id not in selected:
                selected.append(node_id)
                break
    return selected


def main() -> None:
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, Phase0TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    resolve_defaults(data_args, model_args, training_args)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    transformers.set_seed(training_args.seed)
    random.seed(data_args.phase0_seed)
    torch.manual_seed(data_args.phase0_seed)
    _install_accelerate_dispatch_batches_compat()

    if "checkpoint-10200" in str(model_args.pretrain_mm_mlp_adapter):
        raise ValueError(
            "Legacy Products checkpoint-10200 is a representation-trigger attack and is forbidden; "
            f"got {model_args.pretrain_mm_mlp_adapter}"
        )
    required_paths = [data_args.train_jsonl, data_args.candidate_pool, model_args.pretrain_mm_mlp_adapter]
    if data_args.trigger_init_json:
        required_paths.append(data_args.trigger_init_json)
    if data_args.trigger_logits_init_path:
        required_paths.append(data_args.trigger_logits_init_path)
    for required in required_paths:
        if not Path(required).exists():
            raise FileNotFoundError(required)
    model_args.mm_hidden_size = 2432 + 111
    compute_dtype = torch.float16 if training_args.fp16 else torch.bfloat16 if training_args.bf16 else torch.float32

    model = LlagaLlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=compute_dtype,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model.get_model().initialize_graph_modules(model_args=model_args, fsdp=training_args.fsdp)
    model.config.tune_mm_mlp_adapter = True
    model.config.mm_use_graph_start_end = model_args.mm_use_graph_start_end
    model.config.mm_use_graph_patch_token = model_args.mm_use_graph_patch_token
    model.requires_grad_(False)
    for parameter in model.get_model().mm_projector.parameters():
        parameter.requires_grad = True

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    tokenizer = _from_pretrained_tokenizer_with_legacy_fallback(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        local_files_only=True,
    )
    tokenizer.pad_token = tokenizer.unk_token
    conversation_lib.default_conversation = conversation_lib.conv_templates["v1"]
    model.initialize_graph_tokenizer(model_args, tokenizer=tokenizer)

    dataset = Phase0Dataset(tokenizer, data_args)
    if data_args.projector_phase_steps < 0 or data_args.trigger_phase_steps < 0:
        raise ValueError("projector_phase_steps and trigger_phase_steps must be non-negative")
    if data_args.projector_phase_steps == 0 and data_args.trigger_phase_steps == 0:
        raise ValueError("at least one of projector_phase_steps and trigger_phase_steps must be positive")
    if data_args.trigger_temperature_decay_steps < 0:
        raise ValueError("trigger_temperature_decay_steps must be non-negative")
    if data_args.lambda_clean_distill < 0:
        raise ValueError("lambda_clean_distill must be non-negative")
    if data_args.distill_temperature <= 0:
        raise ValueError("distill_temperature must be positive")
    initial_trigger_logits = torch.randn(NUM_TRIGGER_SLOTS, len(dataset.candidate_ids)) * 0.01
    initial_trigger_node_ids: Optional[list[int]] = None
    if data_args.trigger_logits_init_path:
        initial_trigger_logits = load_trigger_logits(Path(data_args.trigger_logits_init_path))
        expected_shape = (NUM_TRIGGER_SLOTS, len(dataset.candidate_ids))
        if tuple(initial_trigger_logits.shape) != expected_shape:
            raise ValueError(
                "trigger_logits_init_path shape mismatch: "
                f"expected {expected_shape}, got {tuple(initial_trigger_logits.shape)}"
            )
        initial_trigger_node_ids = unique_top1_node_ids(
            initial_trigger_logits,
            dataset.candidate_ids,
        )
        if data_args.trigger_init_json:
            init_payload = json.loads(Path(data_args.trigger_init_json).read_text(encoding="utf-8"))
            declared_node_ids = [
                int(value)
                for value in init_payload.get("selected_node_ids", init_payload.get("trigger_node_ids", []))
            ]
            if declared_node_ids != initial_trigger_node_ids:
                raise ValueError(
                    "trigger_init_json does not match trigger_logits_init_path top-1 IDs: "
                    f"declared={declared_node_ids}, logits={initial_trigger_node_ids}"
                )
    elif data_args.trigger_init_json:
        init_payload = json.loads(Path(data_args.trigger_init_json).read_text(encoding="utf-8"))
        initial_trigger_node_ids = [
            int(value)
            for value in init_payload.get("selected_node_ids", init_payload.get("trigger_node_ids", []))
        ]
        if len(initial_trigger_node_ids) != NUM_TRIGGER_SLOTS or len(set(initial_trigger_node_ids)) != NUM_TRIGGER_SLOTS:
            raise ValueError(
                "trigger_init_json must provide four unique selected_node_ids, "
                f"got {initial_trigger_node_ids}"
            )
        candidate_index = {node_id: index for index, node_id in enumerate(dataset.candidate_ids)}
        if any(node_id not in candidate_index for node_id in initial_trigger_node_ids):
            raise ValueError(
                "trigger_init_json contains a node outside the candidate pool: "
                f"{initial_trigger_node_ids}"
            )
        initial_trigger_logits.zero_()
        for slot, node_id in enumerate(initial_trigger_node_ids):
            initial_trigger_logits[slot, candidate_index[node_id]] = float(data_args.trigger_init_bias)
    model.trigger_logits = nn.Parameter(initial_trigger_logits.float())
    projector_only = data_args.trigger_phase_steps <= 0
    if projector_only:
        model.trigger_logits.requires_grad_(False)
    trainer = Phase0Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        data_collator=Phase0Collator(tokenizer),
        data_args=data_args,
    )
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    if projector_only:
        # The hard projector run receives graphs whose placeholders have
        # already been replaced by real node IDs.  Its trigger logits are
        # freshly initialized and frozen, so selecting from them would emit
        # meaningless random metadata.  Preserve the trigger that is actually
        # present in the hard training JSONL instead.
        selected: Optional[list[int]] = None
        for row in dataset.list_data_dict:
            hard_ids = row.get("poison_meta", {}).get("hard_trigger_node_ids")
            if hard_ids is None:
                continue
            current = [int(node_id) for node_id in hard_ids]
            if len(current) < 2 or len(set(current)) != len(current):
                raise ValueError(
                    "hard_trigger_node_ids must contain at least two unique node IDs, "
                    f"got {current}"
                )
            if selected is None:
                selected = current
            elif selected != current:
                raise ValueError(
                    "inconsistent hard_trigger_node_ids across training JSONL: "
                    f"{selected} vs {current}"
                )
        if selected is None:
            raise ValueError(
                "projector-only mode requires poison_meta.hard_trigger_node_ids "
                "in the hard training JSONL"
            )
        probabilities = None
        selection_source = "hard_trigger_node_ids_from_training_jsonl"
        selection_temperature = None
        selection_is_trained = False
    else:
        probabilities = F.softmax(
            model.trigger_logits.detach().float() / data_args.min_temperature, dim=-1
        )
        selected = []
        for slot in range(NUM_TRIGGER_SLOTS):
            for candidate_index in torch.argsort(probabilities[slot], descending=True).tolist():
                node_id = dataset.candidate_ids[candidate_index]
                if node_id not in selected:
                    selected.append(node_id)
                    break
        selection_source = "trained_trigger_logits_top1_unique"
        selection_temperature = data_args.min_temperature
        selection_is_trained = True
    output_dir = Path(training_args.output_dir)
    torch.save(model.trigger_logits.detach().cpu(), output_dir / "trigger_logits.pt")
    (output_dir / "hardened_trigger.json").write_text(
        json.dumps(
            {
                "protocol": (
                    "phase0_hard_projector_fixed_trigger"
                    if projector_only
                    else "phase0_top1_hardening_unique"
                ),
                "candidate_pool": str(Path(data_args.candidate_pool).resolve()),
                "candidate_node_ids": dataset.candidate_ids,
                "selected_node_ids": selected,
                "selection_source": selection_source,
                "selection_is_trained": selection_is_trained,
                "initial_trigger_node_ids": initial_trigger_node_ids,
                "trigger_learning_rate": data_args.trigger_learning_rate,
                "trigger_init_bias": data_args.trigger_init_bias,
                "trigger_logits_init_path": (
                    str(Path(data_args.trigger_logits_init_path).resolve())
                    if data_args.trigger_logits_init_path
                    else None
                ),
                "trigger_selection_mode": data_args.trigger_selection_mode,
                "gumbel_seed": (
                    data_args.gumbel_seed
                    if data_args.gumbel_seed >= 0
                    else data_args.phase0_seed
                ),
                "projector_gumbel_seed": (
                    data_args.projector_gumbel_seed
                    if data_args.projector_gumbel_seed >= 0
                    else (
                        data_args.gumbel_seed
                        if data_args.gumbel_seed >= 0
                        else data_args.phase0_seed
                    )
                    + 104729
                ),
                "trigger_warmup_steps": data_args.trigger_warmup_steps,
                "trigger_temperature_decay_steps": data_args.trigger_temperature_decay_steps,
                "temperature_for_selection": selection_temperature,
                "probabilities": None if probabilities is None else probabilities.tolist(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output_dir), "selected_node_ids": selected}, indent=2))


if __name__ == "__main__":
    main()
