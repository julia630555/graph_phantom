# Make it more memory efficient by monkey patching the LLaMA model with FlashAttn.
# Need to call this before importing transformers.
#
import os
import sys
from pathlib import Path

GRAPHGPT_CODE_ROOT = Path(__file__).resolve().parents[2]
if str(GRAPHGPT_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(GRAPHGPT_CODE_ROOT))

# GraphGPT upstream assumes an older flash-attn API. In newer environments the
# monkey patch import can fail; fallback to vanilla attention so training can
# still start. Also allow explicit opt-out by env var for k-bit runs.
disable_flash_attn = os.environ.get("GRAPHGPT_DISABLE_FLASH_ATTN", "0") == "1"
if not disable_flash_attn:
    try:
        from graphgpt.train.llama_flash_attn_monkey_patch import (
            replace_llama_attn_with_flash_attn,
        )

        replace_llama_attn_with_flash_attn()
    except Exception as e:
        print(f"[WARN] FlashAttn monkey patch disabled: {e}")
else:
    print("[train_mem] FlashAttention patch disabled by GRAPHGPT_DISABLE_FLASH_ATTN=1")

from graphgpt.train.train_graph import train

if __name__ == "__main__":
    train()
