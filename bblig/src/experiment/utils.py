import wandb
import torch
import warnings
import omegaconf
from pathlib import Path
from omegaconf import DictConfig

def extract_output_dir(config: DictConfig) -> Path:
    '''
    Extracts path to output directory created by Hydra as pathlib.Path instance
    '''
    date = '/'.join(list(config._metadata.resolver_cache['now'].values()))
    output_dir = Path.cwd() / 'outputs' / date
    return output_dir

def preprocess_config(config):
    config.exp.log_dir = extract_output_dir(config)

def setup_wandb(config):
    group, name = str(config.exp.log_dir).split('/')[-2:]
    wandb_config = omegaconf.OmegaConf.to_container(
        config, resolve = True, throw_on_missing = True)
    wandb.init(
        project = config.wandb.project,
        dir = config.exp.log_dir,
        group = group,
        name = name,
        config = wandb_config)

def _normalize_scale(attr: torch.tensor, scale_factor: float):
    assert torch.all(scale_factor != 0), "Cannot normalize by scale factor = 0"
    if torch.any(scale_factor.abs() < 1e-5):
        warnings.warn(
            "Attempting to normalize by value approximately 0, visualized results"
            "may be misleading. This likely means that attribution values are all"
            "close to 0."
        )
    scale_factor = scale_factor.view(-1, 1, 1)
    assert len(attr.shape) == 3
    attr_norm = attr / scale_factor
    return torch.clip(attr_norm, -1, 1)

def normalize_attr(
        attr: torch.tensor, 
        sign: str, 
        outlier_perc: int = 2,
        reduction_axis: int = 1):
    
    def _cumulative_sum_threshold(values: torch.tensor, percentile: int):
        # given values should be non-negative
        assert percentile >= 0 and percentile <= 100, (
            "Percentile for thresholding must be " "between 0 and 100 inclusive."
        )
        batch_size = values.shape[0]
        sorted_vals = values.view(batch_size, -1).sort(dim = 1)[0]
        cum_sums = sorted_vals.cumsum(dim = 1)
        threshold_id = (cum_sums > cum_sums[:, -1, None] * 0.01 * percentile).int().argmax(dim = 1)
        return torch.gather(sorted_vals, 1, threshold_id[:, None]).flatten()

    attr_combined = attr.sum(dim = reduction_axis)

    # Choose appropriate signed values and rescale, removing given outlier percentage
    if sign == 'all':
        threshold = _cumulative_sum_threshold(attr_combined.abs(), 100 - outlier_perc)
    elif sign == 'positive':
        attr_combined = (attr_combined > 0) * attr_combined
        threshold = _cumulative_sum_threshold(attr_combined, 100 - outlier_perc)
    elif sign == 'negative':
        attr_combined = (attr_combined < 0) * attr_combined
        threshold = -1 * _cumulative_sum_threshold(attr_combined.abs(), 100 - outlier_perc)
    elif sign == 'absolute_value':
        attr_combined = attr_combined.abs()
        threshold = _cumulative_sum_threshold(attr_combined, 100 - outlier_perc)
    else:
        raise AssertionError("Visualize sign type is not valid.")
    
    return _normalize_scale(attr_combined, threshold)