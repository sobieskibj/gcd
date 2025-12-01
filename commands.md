### GCD ImageNet (zebra ↔ sorrel) — quick commands

Prereqs
- You already trained a DiffAE autoencoder at 256×256 and have its checkpoint path (last.ckpt).
- You have a 256×256 source image (.png/.jpg) for zebra or sorrel.
- Optional: a custom ImageNet classifier checkpoint (otherwise torchvision ResNet-50 pretrained is used).

Setup (one-time)
```bash
cd /Users/mgalkowski/Desktop/diffae/gcd
conda create --name gcd python=3.11 -y
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
```

Zebra example (label 340):

```bash
python src/main.py \
  --config-path ../configs/single_image_gmc_mlp_proxy_training/imagenet/resnet/zebra \
  --config-name config \
  strategy.src_img_path=null \
  strategy.hf_dataset_name=imagenet-1k \
  strategy.hf_split=train \
  strategy.hf_label=340 \
  strategy.hf_index=0 \
  device=cuda:0 \
  wandb.project=null
```

Run: sorrel (class 339, sorrel→not‑sorrel direction)
```bash
cd /Users/mgalkowski/Desktop/diffae/gcd/gcd
python src/main.py \
  --config-path ../configs/single_image_gmc_mlp_proxy_training/imagenet/resnet/sorrel \
  --config-name config \
  strategy.src_img_path="$SRC_IMG" \
  strategy.dae_kwargs.path_ckpt="$DAE_CKPT" \
  device=cuda:0 \
  wandb.project=null
```

Notes
- The configs use the default 3‑channel DiffAE wrapper and a ResNet‑50 ImageNet classifier.
- To use a custom classifier, add at the end of the command:
  - `strategy.ce_loss_kwargs.components.clf.path_to_weights="/path/to/classifier.ckpt"`
- Outputs are written under `gcd/gcd/outputs/<date_time>/strategy/`. Proxy data and candidate directions are in `strategy/proxy/...`.

Direction transfer (apply a saved direction to another image)
```bash
# Example placeholders — update paths after a run completes
export LOG_DIR="/Users/mgalkowski/Desktop/diffae/gcd/gcd/outputs/<date_time>/strategy"
export DIR_PATH="$LOG_DIR/proxy/<some_subdir>/<direction_file>.pt"    # e.g., *_grad_*.pt
export TARGET_IMG="/absolute/path/to/another_image.png"

python src/direction_transfer.py \
  --log-dir-path "$LOG_DIR" \
  --direction-path "$DIR_PATH" \
  --img-path "$TARGET_IMG" \
  --subdir-name "transfer_run_1"
```

```bash
python src/direction_transfer_hf_batch.py \
  --direction-path "/path/to/outputs/<date_time>/proxy/0/0_grad_*.pt" \
  --log-dir-path "/path/to/outputs/<date_time>" \
  --dataset-name imagenet-1k \
  --split train \
  --target-class 340 \
  --n-samples 500 \
  --step-size 1.0 \
  --T-render 100 \
  --subdir-name "hf_batch_zebra"
```

Tips for memory/time
- If you see OOM, reduce: `strategy.dae_kwargs.batch_size` (e.g., 128) or narrow `strategy.dae_kwargs.std`.
- To speed up renders, you can lower `strategy.dae_kwargs.T_render` (e.g., 20–50).

Switching class direction
- Zebra run uses `query_label=340`; sorrel run uses `query_label=339`. Use the matching config.

