import os
import random
import numpy as np
import torch

import logging
log = logging.getLogger(__name__)

def set_seed(seed):
    log.info(f'Seed set to {seed}')
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def split_grid(grid, image_res = 256, padding_width = 2):
    imgs = []
    n_cols = int(grid.shape[2] / (image_res + padding_width))
    n_rows = int(grid.shape[1] / (image_res + padding_width))
    for row_id in range(n_rows):
        for col_id in range(n_cols):
            idx_0 = padding_width + (image_res + padding_width) * row_id
            idx_1 = padding_width + (image_res + padding_width) * col_id
            img = grid[:, idx_0:idx_0 + image_res, idx_1:idx_1 + image_res]
            imgs.append(img)
    return imgs