# Small LLM: shop assistant fine-tune

<p align="center">
  <strong>Fine-tune Qwen3 0.6B on a laptop GPU to turn a customer message into the next shop action as JSON.</strong>
</p>

<p align="center">
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.10-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"></a>
  <a href="https://huggingface.co/Qwen"><img src="https://img.shields.io/badge/Qwen3-0.6B%20base-6F42C1?style=for-the-badge" alt="Qwen3"></a>
  <a href="https://docs.astral.sh/uv/"><img src="https://img.shields.io/badge/uv-package%20manager-DE5FE9?style=for-the-badge&logo=uv&logoColor=white" alt="uv"></a>
</p>

A small model does not need to know the catalog, prices, or stock. Those stay in tools. Fine-tuning teaches it the **format** and the **store's action set**, so a 0.6B model can sit in front of real tools and route a conversation reliably and cheaply.

## Before / after

Customer (Spanish): *"Busco tenis para correr talla 42"*

| | Output |
|---|---|
| **Base model** | `¿Qué tipo de tenis prefieres?` |
| **Fine-tuned** | `{"action":"search_products","slots":{"category":"running shoes","size":"42","color":"no"},"missing":[],"reply":"Buscando tenis para correr, talla 42."}` |

## How it works

| Step | Notebook | What happens |
|---|---|---|
| 1. Dataset | [`shop_assistant_dataset.ipynb`](notebooks/shop_assistant_dataset.ipynb) | Synthetic generator (template banks + small catalogs, fixed seed, no LLM calls). It produces instruction rows whose output is JSON with `action`, `slots`, `missing`, `reply`. Each row is validated, then split 85 / 10 / 5. |
| 2. Fine-tune | [`finetuning.ipynb`](notebooks/finetuning.ipynb) | Loads Qwen3 0.6B base weights in plain PyTorch. Trains 218.5M of 751.6M parameters with AdamW and gradient accumulation, with the loss masked to the response. |
| 3. Evaluate | [`finetuning.ipynb`](notebooks/finetuning.ipynb) | Greedy decoding with stop sequences, before/after comparisons, and diagrams of which weights moved. |

Trained on an **RTX 3060 Laptop GPU (6 GB)**.

## Parameter-efficient variants

Each notebook stands on its own. It uses the same model, dataset, loss mask and training loop as `finetuning.ipynb`, and swaps out only what gets trained. The adapters are written from scratch in PyTorch (no `peft`, no `bitsandbytes`) and go on all 7 linear layers of all 28 blocks.

| Notebook | Technique | What is trained | Trainable params (Qwen3 0.6B) |
|---|---|---|---|
| [`finetuning.ipynb`](notebooks/finetuning.ipynb) | Partial finetune | Last 4 blocks, final norm, output head | 218.5M |
| [`lora.ipynb`](notebooks/lora.ipynb) | LoRA | `ΔW = B A`, rank 8; merged back into `W` after training | 5.05M |
| [`qlora.ipynb`](notebooks/qlora.ipynb) | QLoRA | LoRA on a base quantized to 4-bit NF4 (hand-written quantizer) | 5.05M |
| [`lora_fa.ipynb`](notebooks/lora_fa.ipynb) | LoRA-FA | Only `B`; `A` stays frozen and random, which saves activation memory | 2.75M |
| [`tinylora.ipynb`](notebooks/tinylora.ipynb) | TinyLoRA | One vector `v` shared by every layer: `ΔW = U Σ (Σ vᵢ Pᵢ) Vᵀ` | 13 |

## Run it

```bash
uv sync
uv run jupyter lab notebooks/
```

Run `shop_assistant_dataset.ipynb` first, then `finetuning.ipynb` or any of the variant notebooks. The first run downloads the Qwen3 0.6B base weights into `qwen3/`. Fine-tuning needs a CUDA GPU.

## Credits

The Qwen3 model and tokenizer code in [`reasoning_from_scratch/qwen3.py`](reasoning_from_scratch/qwen3.py) is adapted from Sebastian Raschka's [reasoning-from-scratch](https://github.com/rasbt/reasoning-from-scratch) (Apache-2.0).
