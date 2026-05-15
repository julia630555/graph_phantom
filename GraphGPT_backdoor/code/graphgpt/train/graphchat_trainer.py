import os
import json
import torch
import torch.nn as nn

from transformers import Trainer
from typing import Dict, Optional, Sequence


def unwrap_model(model: nn.Module) -> nn.Module:
    """
    Recursively unwraps a model from potential containers (as used in distributed training).

    Args:
        model (`torch.nn.Module`): The model to unwrap.
    """
    # since there could be multiple levels of wrapping, unwrap recursively
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    else:
        return model


class GraphChatTrainer(Trainer):

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        model_to_save = unwrap_model(self.model)
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
        save_graph_projector = getattr(self.args, 'tune_graph_mlp_adapter', False)
        save_graph_tower = (
            getattr(self.args, 'tune_graph_tower', False)
            or getattr(self.args, 'save_graph_tower_adapter', False)
        )

        # Save trigger package when present.
        trigger_module = None
        if hasattr(model_to_save, "get_model"):
            trigger_module = getattr(model_to_save.get_model(), "backdoor_trigger", None)

        adapter_state_dict = None
        if save_graph_projector or save_graph_tower:
            adapter_state_dict = state_dict
            if adapter_state_dict is None:
                # Only save the model itself if we are using distributed training
                adapter_state_dict = model_to_save.state_dict()

        if save_graph_projector:
            # Save the model
            weight_to_save = {}
            for k, v in adapter_state_dict.items():
                if 'graph_projector' in k:
                    weight_to_save[k] = v
                elif trigger_module is None and ('embed_tokens' in k or 'embed_in' in k):
                    weight_to_save[k] = v
                elif trigger_module is not None and k == 'model.embed_tokens.weight':
                    # v20 keeps token embeddings frozen. Store only the two graph
                    # start/end rows needed to reproduce the clean adapter state.
                    # Clone the slice so torch.save does not retain the full
                    # embedding storage behind this tiny view.
                    weight_to_save[k] = v[-2:].detach().cpu().clone()

            current_folder = output_dir.split('/')[-1]
            parent_folder = os.path.dirname(output_dir)
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "graph_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'graph_projector.bin'))

        if save_graph_tower:
            weight_to_save = {
                k: v
                for k, v in adapter_state_dict.items()
                if 'graph_tower' in k
            }
            if not weight_to_save and hasattr(model_to_save, "get_graph_tower"):
                graph_tower_module = model_to_save.get_graph_tower()
                if graph_tower_module is not None:
                    weight_to_save = graph_tower_module.state_dict()

            current_folder = output_dir.split('/')[-1]
            parent_folder = os.path.dirname(output_dir)
            if current_folder.startswith('checkpoint-'):
                graph_tower_folder = os.path.join(parent_folder, "graph_tower")
                os.makedirs(graph_tower_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(graph_tower_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'graph_tower.bin'))

        if trigger_module is not None and output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            torch.save(trigger_module.state_dict(), os.path.join(output_dir, "trigger_state.pt"))
            cfg_meta = {}
            if hasattr(trigger_module, "meta_dict"):
                cfg_meta = trigger_module.meta_dict()
            elif hasattr(trigger_module, "cfg"):
                cfg_meta = vars(trigger_module.cfg)
            with open(os.path.join(output_dir, "trigger_meta.json"), "w", encoding="utf-8") as f:
                json.dump(cfg_meta, f, indent=2, ensure_ascii=False)

        if save_graph_projector or save_graph_tower:
            if output_dir is not None:
                os.makedirs(output_dir, exist_ok=True)
                if hasattr(model_to_save, "config"):
                    model_to_save.config.save_pretrained(output_dir)
                if getattr(self, "tokenizer", None) is not None:
                    self.tokenizer.save_pretrained(output_dir)
            return

        # transformers does not support save_pretrained() for k-bit models.
        # In that case we keep adapter/config/tokenizer artifacts and skip the
        # full-model save path.
        if getattr(model_to_save, "is_quantized", False):
            if output_dir is not None:
                os.makedirs(output_dir, exist_ok=True)
                if hasattr(model_to_save, "config"):
                    model_to_save.config.save_pretrained(output_dir)
                if getattr(self, "tokenizer", None) is not None:
                    self.tokenizer.save_pretrained(output_dir)
            return

        super(GraphChatTrainer, self)._save(output_dir, state_dict)
