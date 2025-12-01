import sys
import os
import torch
import pandas as pd
import torchvision
import omegaconf
from pathlib import Path
from argparse import ArgumentParser
from datasets import load_dataset
from PIL import Image
import torchvision.transforms as T
from typing import Optional, Dict

from proxies import *
from losses import *
from classifiers import *
from dae import *

import logging
logging.basicConfig(level = logging.INFO)
log = logging.getLogger(__name__)

#
# Fill these with your paths if you want per-split defaults instead of passing --predictions-csv
#
PREDICTIONS_PATHS: Dict[str, Dict[str, Optional[str]]] = {
    # Example:
    # 'imagenet-1k': {
    #     'train': '/abs/path/to/imagenet1k_train_predictions.csv',
    #     'validation': '/abs/path/to/imagenet1k_val_predictions.csv',
    # },
}

def init_class_from_string(class_name):
    return getattr(sys.modules[__name__], class_name)

def get_cfg(args):
    path_cfg = (Path(args.log_dir_path) if args.log_dir_path else Path(args.direction_path).parents[2]) / '.hydra' / 'config.yaml'
    cfg = omegaconf.OmegaConf.load(path_cfg)
    return cfg

def init_loss(cfg, args):
    loss_kwargs = cfg.strategy.ce_loss_kwargs
    loss_kwargs['make_output_dir'] = False
    clf_kwargs = loss_kwargs.components.clf
    clf_name = clf_kwargs.pop('_target_').split('.')[-1]
    log.info(f'Using {clf_name} classifier')
    if clf_name == 'DenseNet':
        clf_kwargs.use_probs_and_query_label = False
    elif clf_name == 'ResNet':
        clf_kwargs.use_softmax_and_query_label = True
    clf = init_class_from_string(clf_name)(**clf_kwargs)
    # Override target/query class if provided for this direction
    if getattr(args, "direction_target_class", None) is not None:
        if hasattr(clf, "query_label"):
            clf.query_label = int(args.direction_target_class)
            log.info(f"Set classifier query_label to direction target class: {clf.query_label}")
    comps_kwargs = loss_kwargs.components
    comps_name = comps_kwargs._target_.split('.')[-1]
    comps_kwargs.pop('_target_')
    comps_kwargs.pop('clf', None)
    comps_kwargs.pop('src_img_path', None)
    comps = init_class_from_string(comps_name)(
        src_img_path = None,
        clf = clf,
        **comps_kwargs)
    loss_kwargs.pop('components')
    loss_name = 'CounterfactualLossFromGeneralComponents'
    loss_kwargs.weight_cls = cfg.strategy.ce_loss_kwargs.weight_cls
    loss_kwargs.weight_lpips = cfg.strategy.ce_loss_kwargs.weight_lpips
    loss = init_class_from_string(loss_name)(components = comps, **loss_kwargs)
    return loss

def init_dae(cfg):
    dae_kwargs = cfg.strategy.dae_kwargs
    dae_kwargs['make_output_dir'] = False
    dae_type = cfg.strategy.dae_type
    if dae_type == 'default':
        dae_class = DAE
    elif dae_type == 'chexpert':
        dae_class = DAECheXpert
    else:
        raise NotImplementedError('DAE type not recognized')
    log.info(f'Using {dae_class}')
    dae = dae_class(**dae_kwargs)
    return dae

def load_direction(path, device):
    grad = torch.load(Path(path), map_location='cpu').to(device)
    if torch.count_nonzero(grad) == 0:
        raise ValueError("Direction tensor contains only zeros.")
    return grad

def get_predictions_df(dataset_name: str, split: str, explicit_csv: Optional[str]) -> Optional[pd.DataFrame]:
    """
    Loads predictions CSV expected to have:
      - index column: 'idx'
      - column: 'pred_label'
    If explicit_csv is None, attempts to use PREDICTIONS_PATHS[dataset_name][split].
    Returns None if no path provided.
    """
    csv_path = explicit_csv
    if csv_path is None:
        csv_path = PREDICTIONS_PATHS.get(dataset_name, {}).get(split, None)
    if csv_path is None:
        return None
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Predictions CSV not found: {csv_path}")
    df = pd.read_csv(csv_path, index_col="idx")
    if "pred_label" not in df.columns:
        raise ValueError(f"Predictions CSV missing 'pred_label' column: {csv_path}")
    return df

def _extract_ex_idx(example: dict, fallback: Optional[int] = None) -> Optional[int]:
    """
    Try to extract a stable dataset index from a HF streaming example.
    Prefers '__index_level_0__', then 'id', then 'idx'. Falls back to provided fallback.
    """
    for key in ("__index_level_0__", "id", "idx"):
        if key in example:
            try:
                return int(example[key])
            except Exception:
                continue
    return fallback

def iter_hf_by_label(
    dataset_name,
    split,
    label,
    start_index,
    n_samples,
    token=None,
    cache_dir=None,
    predictions_df: Optional[pd.DataFrame] = None,
    filter_id: Optional[int] = None,
    n_skip: int = 0,
    pred_index_scope: str = "global",  # 'global' or 'label'
):
    kwargs = {"streaming": True}
    if token is None:
        token = os.getenv('HF_TOKEN', None)
    if cache_dir is None:
        cache_dir = os.getenv('DATASET_CACHE', None)
    if token is not None:
        kwargs["token"] = token
    if cache_dir is not None:
        kwargs["cache_dir"] = cache_dir
    ds = load_dataset(dataset_name, split=split, **kwargs)
    count_label_hits = 0           # counts examples with ex['label'] == label (zero-based)
    emitted = 0
    skipped = 0
    for global_idx, ex in enumerate(ds):
        if ex.get('label', None) != label:
            continue
        # Count only matching label occurrences for start_index logic
        current_label_idx = count_label_hits  # zero-based index of label-matching sample
        count_label_hits += 1
        if current_label_idx < start_index:
            continue

        # If predictions filtering is requested, check it here
        if predictions_df is not None and filter_id is not None:
            # Choose index to match predictions: global or label-only enumeration
            ex_idx = global_idx if pred_index_scope == "global" else current_label_idx
            if ex_idx is None or ex_idx not in predictions_df.index:
                # If we can't find a matching prediction idx, skip
                continue
            pred_label = predictions_df.loc[ex_idx, "pred_label"]
            # Keep only correctly predicted examples for the requested class
            # Since ex['label'] == label, correctness means pred_label == label == filter_id
            if int(pred_label) != int(filter_id):
                continue

        # Apply n_skip on the filtered stream
        if skipped < n_skip:
            skipped += 1
            continue

        # Attach derived indices for downstream logging if needed
        ex = dict(ex)
        ex["_global_idx_enumerate"] = global_idx
        ex["_label_idx_enumerate"] = current_label_idx
        yield ex
        emitted += 1
        if emitted >= n_samples:
            break

def pil_to_tensor_01(img: Image.Image, image_size: int):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    transform = T.Compose([
        T.CenterCrop(image_size),
        T.Resize(image_size),
        T.ToTensor(),    # [0,1]
    ])
    return transform(img)

def to_dae_range(x01: torch.Tensor):
    return (x01 - 0.5) * 2

def from_dae_range(xm1p1: torch.Tensor):
    return (xm1p1 + 1) / 2

def main(args):
    log.info("Batch direction transfer (HF ImageNet)")
    cfg = get_cfg(args)
    device = torch.device(cfg.device)
    loss = init_loss(cfg, args)
    dae = init_dae(cfg)
    direction = load_direction(args.direction_path, device)

    out_dir = (Path(args.log_dir_path) if args.log_dir_path else Path(args.direction_path).parents[2]) / 'direction_transfer' / args.subdir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_rows = []

    image_size = dae.config.img_size if hasattr(dae, 'config') else 256
    n_done = 0
    # Optional predictions filtering
    predictions_df = None
    if args.filter_id is not None:
        predictions_df = get_predictions_df(args.dataset_name, args.split, args.predictions_csv)
        if predictions_df is None:
            log.error(f"Filtering requested but no predictions CSV provided for {args.dataset_name}/{args.split}.")
            log.error("Provide --predictions-csv or fill PREDICTIONS_PATHS in the script.")
            sys.exit(1)
    for ex in iter_hf_by_label(
        dataset_name=args.dataset_name,
        split=args.split,
        label=args.target_class,
        start_index=args.start_index,
        n_samples=args.n_samples,
        token=args.hf_token,
        cache_dir=args.hf_cache_dir,
        predictions_df=predictions_df,
        filter_id=args.filter_id,
        n_skip=args.n_skip,
        pred_index_scope=args.pred_index_scope
    ):
        img_pil = ex['image'] if isinstance(ex['image'], Image.Image) else Image.fromarray(ex['image'])
        # Log the index used to match predictions for traceability
        idx = ex.get("_global_idx_enumerate", n_done) if args.pred_index_scope == "global" else ex.get("_label_idx_enumerate", n_done)
        # Prepare tensors
        img01 = pil_to_tensor_01(img_pil, image_size).unsqueeze(0).to(device)
        img_m1p1 = to_dae_range(img01)
        # Encode
        latent_sem = dae.encode(img_m1p1)
        latent_ddim = dae.encode_stochastic(img_m1p1, latent_sem)
        # One-step transfer using fixed step size
        step = torch.tensor([[args.step_size]], device=device)
        grad = direction
        latent_sem_new = latent_sem - step * grad
        img_trans = dae.render(latent_ddim, latent_sem_new, T=args.T_render)
        # Save pair
        orig_name = f"original_{n_done+1}.png"
        trans_name = f"inpaint_{n_done+1}.png"
        torchvision.utils.save_image(img01, out_dir / orig_name)
        torchvision.utils.save_image(img_trans, out_dir / trans_name)
        # Log classifier probabilities (optional)
        with torch.no_grad():
            comps = loss.get_components(img_trans)
            # Query-label probabilities
            prob_after = loss.get_query_label_probability(comps['predictions']).item()
            pred_before_logits = loss.components.classifier(img01)
            prob_before = loss.get_query_label_probability(pred_before_logits).item()
            # Max-arg probabilities and classes (multiclass)
            probs_after_all = torch.softmax(comps['predictions'], dim = 1)
            best_after_prob, best_after_cls = probs_after_all.max(dim = 1)
            probs_before_all = torch.softmax(pred_before_logits, dim = 1)
            best_before_prob, best_before_cls = probs_before_all.max(dim = 1)
        csv_rows.append({
            "idx": int(idx),
            "orig_path": orig_name,
            "inpaint_path": trans_name,
            "prob_before": prob_before,
            "prob_after": prob_after,
            "pred_before_best_cls": int(best_before_cls.item()),
            "pred_before_best_prob": float(best_before_prob.item()),
            "pred_after_best_cls": int(best_after_cls.item()),
            "pred_after_best_prob": float(best_after_prob.item()),
            "direction_target_class": int(getattr(args, "direction_target_class", -1))
        })
        n_done += 1
        if n_done % 25 == 0:
            log.info(f"Processed {n_done}/{args.n_samples}")
    # Save CSV
    pd.DataFrame(csv_rows).to_csv(out_dir / "pairs.csv", index=False)
    log.info(f"Saved {n_done} pairs to {out_dir}")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--direction-path', type=str, required=True, help='Path to direction .pt file from proxy run')
    parser.add_argument('--direction-target-class', type=int, required=True, help='Target class id the direction was trained for (query label)')
    parser.add_argument('--log-dir-path', type=str, default=None, help='Log dir path with .hydra config (defaults to infer from direction path)')
    parser.add_argument('--dataset-name', type=str, default='imagenet-1k', help='HF dataset name')
    parser.add_argument('--split', type=str, default='train', help='HF split')
    parser.add_argument('--target-class', type=int, required=True, help='HF class id to sample')
    parser.add_argument('--n-samples', type=int, default=500, help='Number of images to transfer')
    parser.add_argument('--start-index', type=int, default=0, help='Skip N matching samples before collecting')
    parser.add_argument('--hf-token', type=str, default=None, help='HF token (or set HF_TOKEN env)')
    parser.add_argument('--hf-cache-dir', type=str, default=None, help='HF cache (or set DATASET_CACHE env)')
    parser.add_argument('--step-size', type=float, default=1.0, help='Step size multiplier for the direction')
    parser.add_argument('--T-render', type=int, default=100, help='Number of DDIM steps for rendering')
    parser.add_argument('--subdir-name', type=str, default='hf_batch', help='Output subdir name under direction_transfer')
    # Optional predictions-based filtering (keep only correctly predicted examples for --filter-id)
    parser.add_argument('--filter-id', type=int, default=None, help='If set, keep only samples with this GT label that are correctly predicted as this label')
    parser.add_argument('--n-skip', type=int, default=0, help='Skip first N samples after filtering (useful for sharding)')
    parser.add_argument('--predictions-csv', type=str, default=None, help='Optional override path to predictions CSV (index=idx, col=pred_label)')
    args = parser.parse_args()
    main(args)

