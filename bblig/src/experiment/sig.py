import torch
import wandb
import lightning as L
import pandas as pd
import torch.nn.functional as F
from torchvision.transforms.functional import convert_image_dtype
from tqdm import tqdm
from omegaconf import DictConfig
from hydra.utils import instantiate
from torchvision.utils import make_grid

from .utils import preprocess_config, setup_wandb, normalize_attr

import logging
log = logging.getLogger(__name__)

SIGNS = ['positive', 'negative', 'absolute_value']

def run(config: DictConfig):
    preprocess_config(config)
    setup_wandb(config)

    log.info('Launching Fabric')
    fabric = get_fabric(config)
    
    log.info('Building components')
    classifier, dae, gcd, evaluator = get_components(config, fabric)

    log.info('Initializing dataloader')
    dataloader = get_dataloader(config, fabric)

    log.info(f'Starting experiment')
    for idx, batch in enumerate(dataloader):
        log.info(f'Batch: {idx}')
        batch_imgs, batch_idx = batch
        log_imgs(batch_imgs)

        ## 0. get classifier predictions
        log.info("Computing classifier's predictions")
        with torch.no_grad():
            batch_probs_orig = classifier.pred_prob(batch_imgs)[:, config.exp.target_id]
        log_probs_orig(batch_idx, batch_probs_orig)

        ## 1. encode img into z_sem and z_diff
        log.info('Encoding images into latent space')
        batch_z_sem = dae.encode(batch_imgs)
        batch_z_diff = dae.encode_stochastic(batch_imgs, batch_z_sem)

        ## 2. traverse z_sem along the provided gcd
        log.info('Traversing semantic code along GCD')
        batch_z_sem_paths, batch_step_sizes, batch_signs = get_z_sem_paths(
            config, fabric, batch_z_sem, batch_probs_orig, gcd)

        ## 3. generate imgs from the gcd path
        log.info('Generating images')
        batch_z_diff_paths = get_z_diff_paths(config, batch_z_diff)
        batch_imgs_paths = dae.render(batch_z_diff_paths, batch_z_sem_paths)

        ## 4. find baseline id
        log.info('Extracting baseline indices')
        batch_baseline_idx = get_baseline_idx(
            config, classifier, batch_imgs_paths, batch_probs_orig)
        del batch_imgs_paths, batch_z_sem_paths # otherwise we encounter OOM

        ## 5. generate the path from original to baseline with a desired length
        log.info('Generating SIG paths')
        # make new step sizes for each image based on baseline id
        batch_step_sizes = get_batch_step_sizes(
            config, fabric, batch_baseline_idx, batch_step_sizes)
        # make new z_sem_paths based on new step sizes
        batch_z_sem_paths = get_z_sem_paths_from_steps(
            config, gcd, batch_z_sem, batch_step_sizes, batch_signs)
        # generate images from new paths
        batch_imgs_paths = get_sig_paths(config, dae, batch_z_sem_paths, batch_z_diff)
        # quantize and dequantize images to denoise them
        batch_imgs_paths = convert_image_dtype(batch_imgs_paths, dtype = torch.uint8)
        batch_imgs_paths = convert_image_dtype(batch_imgs_paths, dtype = torch.float32)
        log_imgs_paths(batch_imgs_paths.split(config.exp.path_length))

        ## 6. compute gradients
        log.info('Computing gradients')
        batch_grads, batch_outputs = get_grads(config, classifier, batch_imgs_paths)
        log_outputs(batch_idx.repeat_interleave(config.exp.path_length), batch_outputs)

        ## 7. integrate
        log.info('Integrating')
        batch_sig, batch_origs, batch_baselines, batch_errors = get_sig(
            config, batch_imgs_paths, batch_grads, batch_outputs)
        log_sig(batch_sig, config.exp.approx_grad)
        log_baselines(batch_baselines)
        log_diffs(batch_origs, batch_baselines)
        log_errors(batch_errors)

        del batch_sig, batch_baselines, batch_errors
        del batch_step_sizes, batch_imgs_paths, batch_grads
        del batch_z_diff_paths, batch_z_sem, batch_z_sem_paths
        del batch_outputs, batch_signs, batch_probs_orig
        del batch_baseline_idx, batch_idx, batch_z_diff
        torch.cuda.empty_cache()

def get_sig_paths(config, dae, batch_z_sem_paths, batch_z_diff):
    path_length = config.exp.path_length
    batch_size_eff = config.exp.n_unique * config.exp.n_steps

    batch_z_sem_paths = batch_z_sem_paths.split(batch_size_eff)
    batch_z_diff = batch_z_diff.repeat_interleave(path_length, 0).split(batch_size_eff)

    batch_imgs_paths = []

    for z_sem_paths, z_diffs in tqdm(zip(batch_z_sem_paths, batch_z_diff)):
        # we check whether baseline was found, if not we append black imgs
        if torch.all(z_sem_paths[0] == z_sem_paths[-1]):
            imgs_paths = torch.zeros_like(z_diffs)
            if dae.config.in_channels == 1:
                imgs_paths = imgs_paths.repeat(1, 3, 1, 1)
            batch_imgs_paths.append(imgs_paths)
        else:
            imgs_paths = dae.render(z_diffs, z_sem_paths)
            batch_imgs_paths.append(imgs_paths)

    return torch.cat(batch_imgs_paths)

def get_z_sem_paths_from_steps(config, gcd, batch_z_sem, batch_step_sizes, batch_signs):
    path_length = config.exp.path_length

    # we use batch_signs to ensure that we move either along gcd or -gcd
    batch_z_sem = batch_z_sem.repeat_interleave(path_length, 0)
    # batch_signs = batch_signs.unsqueeze(1).repeat(path_length, 1)
    # batch_step_sizes = batch_signs * batch_step_sizes.unsqueeze(1)
    batch_step_sizes = batch_step_sizes.unsqueeze(1)
    batch_z_sem_paths = batch_z_sem + batch_step_sizes * gcd.dir

    return batch_z_sem_paths

def get_batch_step_sizes(config, fabric, batch_baseline_idx, batch_step_sizes):
    path_length = config.exp.path_length
    batch_step_sizes_new = []
    # for each image, we create new step sizes based on max_step_size
    # provided by the baseline id
    for baseline_id in batch_baseline_idx:
        max_step_size = batch_step_sizes[baseline_id].item()
        step_sizes = fabric.to_device(torch.linspace(0., max_step_size, path_length))
        batch_step_sizes_new.append(step_sizes)

    return torch.cat(batch_step_sizes_new)
    
def get_baseline_idx(config, classifier, batch_imgs_paths, batch_probs_orig):
    # get classifier's predictions for path
    with torch.no_grad():
        batch_probs = classifier.pred_prob(batch_imgs_paths)[:, config.exp.target_id]

    # get class prediction for original image for probability conversion
    chunk_size = config.exp.n_steps
    batch_pred_class = (batch_probs_orig > 0.5).int()
    batch_pred_class = batch_pred_class.repeat_interleave(chunk_size, 0)
    batch_pred_class = (1 - batch_pred_class).bool()
    
    # convert probs to 1 - probs only if
    # the initial predicted class is 0 to have a unified 
    # range of values where < 0.5 always means that class was flipped
    batch_probs[batch_pred_class] = 1 - batch_probs[batch_pred_class]
    
    # for each image, find baseline id
    chunks_probs = torch.split(batch_probs, chunk_size)
    thr_prob = config.exp.thr_prob
    batch_baseline_idx = []

    for chunk_id, chunk in enumerate(chunks_probs):
        under_thr = chunk < thr_prob
        baseline_id = chunk_id * chunk_size + under_thr.int().argmax()
        batch_baseline_idx.append(baseline_id)
    
    return torch.stack(batch_baseline_idx)

def log_diffs(origs, baselines):
    for orig, baseline in zip(origs, baselines):
        diff = (orig - baseline).sum(0).abs()
        diff -= diff.min()
        diff /= diff.max()
        wandb.log({'diffs': wandb.Image(diff)})

def log_errors(errors):
    for error in errors.numpy(force = True):
        wandb.log({'errors': error})

def log_sig(sigs, approx_grad):

    if approx_grad:

        for sig in sigs:
            dict_to_log = {}
            for sign in SIGNS:
                if torch.all(sig == 0):
                    img = sig
                elif sign == 'positive':
                    img = (sig > 0) * sig
                elif sign == 'negative':
                    img = -1 * (sig < 0) * sig
                elif sign == 'absolute_value':
                    img = sig.abs()
                img -= img.min()
                img /= img.max()
                dict_to_log[f'sigs/{sign}'] = wandb.Image(img)
    else:

        for sig in sigs:
            dict_to_log = {}
            for sign in SIGNS:
                if torch.all(sig == 0):
                    img = wandb.Image(sig)
                else:
                    img = wandb.Image(normalize_attr(sig[None], sign))
                dict_to_log[f'sigs/{sign}'] = img
            dict_to_log['sigs/raw'] = wandb.Image(sig)

    wandb.log(dict_to_log)

def log_baselines(baselines):
    for baseline in baselines:
        wandb.log({'baselines': wandb.Image(baseline)})

def log_outputs(idx, logits):
    data = torch.stack([logits, F.sigmoid(logits)]).T.numpy(force = True)
    df = pd.DataFrame(
         data = data,
         index = idx.numpy(force = True),
         columns = ['logit', 'prob'])
    table = wandb.Table(dataframe = df.reset_index())
    wandb.log({'outputs_paths': table})    

def log_imgs_paths(imgs_paths):
    for chunk in imgs_paths:
        grid = make_grid(chunk.float())
        wandb.log({'imgs_paths': wandb.Image(grid)})

def log_probs_orig(idx, probs):
    df = pd.DataFrame(
         data = probs.numpy(force = True),
         index = idx.numpy(force = True),
         columns = ['prob'])
    table = wandb.Table(dataframe = df.reset_index())
    wandb.log({'probs_orig': table})

def log_imgs(imgs):
    for img in imgs:
        wandb.log({'imgs': wandb.Image(img)})

def get_sig(config, batch_imgs_paths, batch_grads, batch_outputs):
    path_length = config.exp.path_length
    batch_imgs_paths = batch_imgs_paths.split(path_length)

    if config.exp.approx_grad:
        # approx_grad shortens path by 1
        path_length -= 1

    batch_grads = batch_grads.split(path_length)
    
    batch_sigs = []
    batch_origs = []
    batch_baselines = []
    batch_errors = []

    for path_imgs, path_grads in zip(batch_imgs_paths, batch_grads):
        orig = path_imgs[0]
        baseline = path_imgs[-1]

        if torch.all(orig == baseline):
            sig = torch.zeros_like(orig)
            error = torch.full_like(batch_outputs[0], float('inf'))
        else:
            sig = (orig - baseline) * (1 / path_length) * path_grads.sum(0)
            error = sig.sum() - (batch_outputs[0] - batch_outputs[-1])
            sig = sig.sum(0)

        batch_sigs.append(sig)
        batch_origs.append(orig)
        batch_baselines.append(baseline)
        batch_errors.append(error)

    return torch.stack(batch_sigs), torch.stack(batch_origs), torch.stack(batch_baselines), torch.stack(batch_errors)

def get_grads(config, classifier, batch_imgs_paths):
    batch_size_eff = config.exp.n_unique * config.exp.n_steps
    batch_outputs = []
    batch_grads = []

    if config.exp.approx_grad:

        for imgs_paths in batch_imgs_paths.split(config.exp.path_length):
            with torch.no_grad():
                outputs = classifier(imgs_paths)[:, config.exp.target_id]

            df = (outputs[1:] - outputs[:-1]).view(-1, 1, 1, 1)
            dx = imgs_paths[1:] - imgs_paths[:-1]
            grads = df / dx
            grads[dx == 0] = 0.

            batch_grads.append(grads)
            batch_outputs.append(outputs)

    else:

        for imgs_paths in batch_imgs_paths.split(batch_size_eff):

            imgs_paths.requires_grad_()
            outputs = classifier(imgs_paths)[:, config.exp.target_id]
            grads = torch.autograd.grad(torch.unbind(outputs), imgs_paths)[0]

            batch_grads.append(grads)
            batch_outputs.append(outputs)

    return torch.cat(batch_grads), torch.cat(batch_outputs)

def get_z_diff_paths(config, batch_z_diff):
    return batch_z_diff.repeat_interleave(config.exp.n_steps, 0)

def get_z_sem_paths(config, fabric, batch_z_sem, batch_probs, gcd):
    assert gcd.target_id == config.exp.target_id

    # get classifier predictions
    batch_pred_class = (batch_probs > 0.5).int()

    # check agreement with gcd
    # gcd always points to increasing probability
    batch_agree = batch_pred_class == 1
    
    # return -1 if they do agree, i.e. this is when we move 
    # along negative gcd which decreases probability
    # return 1 if they do not agree, i.e. this is when we move
    # along positive gcd which increases probability
    batch_signs = torch.ones_like(batch_pred_class)
    batch_signs[batch_agree] *= -1

    # move batch_z_sem along gcd
    max_step_size = config.exp.max_step_size
    n_steps = config.exp.n_steps
    n_uniq = config.exp.n_unique

    # get proper step sizes and multiply by signs
    batch_step_sizes = fabric.to_device(torch.linspace(0., max_step_size, n_steps))
    batch_step_sizes = batch_step_sizes.unsqueeze(1).repeat(n_uniq, 1)
    batch_signs_rep = batch_signs.repeat_interleave(n_steps).unsqueeze(1)
    batch_step_sizes = batch_signs_rep * batch_step_sizes
    batch_z_sem = batch_z_sem.repeat_interleave(n_steps, 0)

    # moving along
    batch_z_sem_paths = batch_z_sem + batch_step_sizes * gcd.dir

    return batch_z_sem_paths, batch_step_sizes, batch_signs

def get_fabric(config):
    fabric = L.Fabric(**config.fabric)
    fabric.seed_everything(config.exp.seed)
    fabric.launch()
    return fabric

def get_components(config, fabric):
    classifier = fabric.setup(instantiate(config.classifier))
    dae = fabric.setup(instantiate(config.dae))
    # evaluator = fabric.setup(instantiate(config.evaluation)(model = classifier))
    evaluator = None
    gcd = fabric.setup(instantiate(config.gcd))
    return classifier, dae, gcd, evaluator

def get_dataloader(config, fabric):
    return fabric.setup_dataloaders(instantiate(config.dataset))
