import sys
import torch
import pandas as pd
import numpy as np
import torchvision
import omegaconf
from pathlib import Path
from argparse import ArgumentParser

from proxies import *
from losses import *
from classifiers import *
from dae import *

import logging
logging.basicConfig(level = logging.INFO)
log = logging.getLogger(__name__)

def init_class_from_string(class_name):
    '''
    class_name must be a name of a class imported above
    '''
    return getattr(sys.modules[__name__], class_name)

def init_loss(args, cfg):
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
    comps_kwargs = loss_kwargs.components
    comps_name = comps_kwargs._target_.split('.')[-1]
    comps_kwargs.pop('_target_')
    comps_kwargs.pop('clf')
    comps_kwargs.pop('src_img_path')
    comps = init_class_from_string(comps_name)(
        src_img_path = args.img_path,
        clf = clf, 
        **comps_kwargs)
    loss_kwargs.pop('components')
    loss_name = 'CounterfactualLossFromGeneralComponents'
    if args.weight_cls is not None and args.weight_lpips is not None:
        log.info('Using weight_cls and weight_lpips from args')
        weight_cls = args.weight_cls
        weight_lpips = args.weight_lpips
    else:
        log.info('Using weight_cls and weight_lpips from config')
        weight_cls = cfg.strategy.ce_loss_kwargs.weight_cls
        weight_lpips = cfg.strategy.ce_loss_kwargs.weight_lpips
    loss_kwargs.weight_cls = weight_cls
    loss_kwargs.weight_lpips = weight_lpips
    loss = init_class_from_string(loss_name)(components = comps, **loss_kwargs)
    return loss

def init_dae(args, cfg):
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

def get_step_sizes(args, cfg):
    step_sizes = torch.linspace(
        *args.min_max_step_size, 
        args.batch_size,
        device = cfg.device).unsqueeze(1)
    return step_sizes

def get_grad(args, cfg):
    path_grad = Path(args.direction_path)
    grad = torch.load(path_grad, map_location = 'cpu').to(cfg.device)
    return grad

def get_cfg(args):
    path_cfg = get_path_log_dir(args) / '.hydra' / 'config.yaml'
    cfg = omegaconf.OmegaConf.load(path_cfg)
    return cfg

def get_img_latent_sem(args, cfg, dae, img):
    img_latent_sem = dae.encode(img)
    return img_latent_sem

def get_img_latent_ddim(args, cfg, dae, img, latent_sem):
    img_latent_ddim = dae.encode_stochastic(img, latent_sem)
    return img_latent_ddim

def get_img(args, cfg):
    img = torchvision.io.read_image(args.img_path).unsqueeze(0) / 255
    # We scale to [-1, 1] as this is the range required by DAE
    img = (img - 0.5) * 2
    return img.to(cfg.device)

def get_src_img_pred_label(args, cfg, loss):
    path_src_img = get_path_log_dir(args) / 'strategy' / 'src_img.png'
    dae_type = cfg.strategy.dae_type
    if dae_type == 'default':
        read_mode = torchvision.io.ImageReadMode.UNCHANGED
    elif dae_type == 'chexpert':
        read_mode = torchvision.io.ImageReadMode.GRAY
    src_img = torchvision.io.read_image(str(path_src_img), mode = read_mode).unsqueeze(0) / 255
    src_img = src_img.to(cfg.device)
    log.info("Computing classifier's prediction for image used in direction calculation")
    pred_logit = loss.components.classifier(src_img)
    pred_prob = loss.get_query_label_probability(pred_logit)
    src_img_pred_label = 1 if pred_prob > 0.5 else 0
    log.info(f'Class predicted for image used in direction calculation: {src_img_pred_label}')
    log.info(f'Probability: {pred_prob.item()}')
    return src_img_pred_label

def get_path_log_dir(args):
    if args.log_dir_path is None:
        return Path(args.direction_path).parents[2]
    else:
        return Path(args.log_dir_path)

def save_imgs_and_ce_s(cfg, args, batch_line_search_imgs, batch_probs, batch_lpips, step_sizes):
    output_dir = get_path_log_dir(args) / 'direction_transfer' / args.subdir_name
    output_dir.mkdir(parents = True, exist_ok = True)
    torchvision.utils.save_image(batch_line_search_imgs, output_dir / Path(args.img_path).parts[-1])
    ce_ids = batch_probs < 0.5
    df_data = np.vstack([
        step_sizes.numpy(force = True).flatten(),
        batch_probs.numpy(force = True),
        batch_lpips.numpy(force = True).flatten(),
        ce_ids.numpy(force = True)]).T
    df = pd.DataFrame(df_data, columns = ['step_size', 'clf_prob', 'lpips', 'is_ce'])
    df.to_csv(output_dir / 'info.csv')
    img = torchvision.io.read_image(args.img_path).to(cfg.device) / 255
    torchvision.utils.save_image(img, output_dir / 'img_orig.png')
    if torch.any(ce_ids):
        ce_s = batch_line_search_imgs[ce_ids]
        diffs = (ce_s - img.unsqueeze(0)).abs()
        torchvision.utils.save_image(ce_s, output_dir / 'ce_s.png')
        torchvision.utils.save_image(diffs / diffs.amax(dim = (1, 2, 3)).view(-1, 1, 1, 1), output_dir / 'diffs.png')

def save_info(cfg, args, src_img_pred_label, req_img_pred_label):
    output_dir = get_path_log_dir(args) / 'direction_transfer' / args.subdir_name
    with open(output_dir / 'info.txt', 'w') as f:
        f.write(f'Class predicted for image used in direction estimation: {src_img_pred_label}')
        f.write('\n')
        f.write(f'Class predicted for requested image: {req_img_pred_label}')
        f.write('\n')

def main(args):
    log.info('Running')
    log.info(f'Image path: {args.img_path}')
    cfg = get_cfg(args)
    log.info(f'Config: {cfg}')
    log.info(f'Args: {args}')
    loss = init_loss(args, cfg)
    dae = init_dae(args, cfg)
    img = get_img(args, cfg)
    latent_sem = get_img_latent_sem(args, cfg, dae, img)
    if args.random_ddim:
        gen = torch.Generator().manual_seed(args.random_ddim_seed)
        if args.random_ddim_method == 'random':
            latent_ddim = torch.randn(*img.shape, generator = gen).to(cfg.device)
        elif args.random_ddim_method == 'noise':
            latent_ddim_orig = get_img_latent_ddim(args, cfg, dae, img, latent_sem)
            noise = torch.normal(mean = 0., std = 0.1, size = img.shape, generator = gen).to(cfg.device)
            latent_ddim = latent_ddim_orig + noise
    else:
        latent_ddim = get_img_latent_ddim(args, cfg, dae, img, latent_sem)
    grad = get_grad(args, cfg)
    if torch.count_nonzero(grad) == 0:
        log.info('Direction is zero')
        log.info('Exiting')
        exit()
    step_sizes = get_step_sizes(args, cfg)
    src_img_pred_label = get_src_img_pred_label(args, cfg, loss)
    # We compare classifier's predicted label for the source image and
    # the current image. If they differ, we move in the opposite direction.
    # If not, nothing changes.
    if src_img_pred_label != loss.components.clf_pred_label:
        log.info('Moving in reverse direction')
        grad = - grad
    else:
        log.info('Moving in gradient direction')
    batch_latent_sem = latent_sem - step_sizes * grad
    batch_latent_ddim = latent_ddim.repeat(args.batch_size, 1, 1, 1)
    log.info('Generating synthetic images')
    batch_line_search_imgs = dae.render(batch_latent_ddim, batch_latent_sem, args.T_render)
    batch_comps = loss.get_components(batch_line_search_imgs)
    batch_probs = loss.get_query_label_probability(batch_comps['predictions'])
    if loss.components.clf_pred_label == 0:
        batch_probs = 1 - batch_probs
    batch_lpips = batch_comps['lpips']
    save_imgs_and_ce_s(cfg, args, batch_line_search_imgs, batch_probs, batch_lpips, step_sizes)
    save_info(cfg, args, src_img_pred_label, loss.components.clf_pred_label)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--direction-path', type = str, help = 'Path to direction as .pt file')
    parser.add_argument('--img-path', type = str, help = 'Image path')
    parser.add_argument('--log-dir-path', type = str, help = 'Path with .hydra subdir')
    parser.add_argument('--random-ddim', type = bool, default = False, help = 'Whether to use latent DDIM that is random or not')
    parser.add_argument('--random-ddim-seed', type = int, default = 0, help = 'Random seed for creating random latent DDIM')
    parser.add_argument('--random-ddim-method', type = str, default = 'random', help = 'Either "random" (sample DDIM from normal distr) or "noise" (add noise to original DDIM)')
    parser.add_argument('--subdir-name', type = str, required = True, help = 'Name for the subdir created in the log dir of the run from --direction-path.')
    parser.add_argument('--batch-size', type = int, default = 128, help = 'Number of synthetic images in line search')
    parser.add_argument('--T-render', type = int, default = 100, help = 'Number of steps in the denoising process. It must divide 1000 without remainders.')
    parser.add_argument('--min-max-step-size', type = float, nargs = 2, default = [-3., 3.], help = 'Minimum and maximum value \
                        of the coefficient that controls the gradient magnitude. Note that gradient norm is 1 by default.')
    parser.add_argument('--weight-cls', default = 1.0, type = float, help = 'Weight assigned to classifier component of counterfactual loss in gradient computation')
    parser.add_argument('--weight-lpips', default = 0.0, type = float, help = 'Weight assigned to perceptual component of counterfactual loss in gradient computation')
    args = parser.parse_args()
    main(args)