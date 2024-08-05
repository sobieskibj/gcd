import torch
import torch.nn as nn
from collections import OrderedDict
from pathlib import Path
from typing import Union, List

from .renderer import render_condition
from .utils import make_config

import logging
log = logging.getLogger(__name__)

class DAECheXpert(nn.Module):
    """
    DiffAE stripped from pl.LightningModule dependency and wrapped
    for inference purposes only. Adapted to CheXpert single-channel data.
    """

    def __init__(
            self, 
            config_name: str,
            batch_size: int,
            std: Union[float, List[float]],
            T_encode: int,
            T_render: int,
            path_ckpt: str,
            device: str,
            output_dir: str,
            distribution: str = 'normal',
            max_norm: float = 1.,
            make_output_dir: bool = True):
        """
        batch_size - size of the batch containing perturbed latent representations 
            of source image
        T_encode - number of diffusion steps for encoding
        T_render - number of diffusion steps for rendering
        distribution - probability distribution used in creating perturbed latent representations.
            Can be either:
                'normal' - samples from the normal distribution centered at src_latent_sem
                'uniform_norm' - samples vectors with norm distributed uniformly over the
                    [0., max_norm] interval
        max_norm - used if distribution is 'uniform_norm'.
        std - used if distribution is 'normal'. Standard deviation(s) of the normal distribution
            used in generating perturbed latent representations
        """
        super(DAECheXpert, self).__init__()
        self.output_dir = Path(output_dir)
        if make_output_dir:
            self.output_dir.mkdir(parents = True)
        self.device = torch.device(device)
        self.config = make_config(config_name)
        self.batch_size = batch_size
        if not isinstance(std, float):
            self.std = torch.tensor(std)
        else:
            self.std = torch.tensor([std])
        self.sampler_encode = self.config._make_diffusion_conf(T_encode).make_sampler()
        self.sampler_render = self.config._make_diffusion_conf(T_render).make_sampler()
        self.distribution = distribution
        self.max_norm = max_norm
        self.make_model(path_ckpt)
        self.to(self.device)
        self.eval()

    def make_model(self, path_ckpt):
        model = self.config.make_model_conf().make_model()
        state_dict = torch.load(path_ckpt, map_location = 'cpu')['state_dict']
        # We only load the ema unet + encoder part of the model
        ema_model_state_dict = OrderedDict(
            [('.'.join(k.split('.')[1:]), v) for k, v in state_dict.items() if 'ema_model' in k])
        model.load_state_dict(ema_model_state_dict)
        model.requires_grad_(False)
        model.eval()
        self.model = model

    def forward(self, x):
        # This is just for compatibility with nn.Module and is rather not used
        return self.model(x)

    @torch.no_grad()
    def encode(self, x):
        assert self.config.model_type.has_autoenc()
        cond = self.model.encoder.forward(x)
        return cond

    @torch.no_grad()
    def encode_stochastic(self, x, cond, T = None):
        if T is not None:
            sampler = self.config._make_diffusion_conf(T).make_sampler()
        else:
            sampler = self.sampler_encode
        out = sampler.ddim_reverse_sample_loop(self.model,
                                               x,
                                               model_kwargs = {'cond': cond})
        return out['sample']
    
    @torch.no_grad()
    def render(self, noise, cond, T = None):
        if T is not None:
            sampler = self.config._make_diffusion_conf(T).make_sampler()
        else:
            sampler = self.sampler_render
        pred_img = render_condition(self.config,
                                    self.model,
                                    noise,
                                    sampler = sampler,
                                    cond = cond)
        pred_img = (pred_img + 1) / 2
        return pred_img

    def make_batch_latent_sem(self, latent_sem):
        batch_latent_sem = latent_sem.repeat(self.batch_size, 1)
        if self.distribution == 'normal':
            noises = []
            log.info(f'Sampling latent sem perturbations from normal distribution with std from {self.std}')
            chunk_size = self.batch_size // len(self.std)
            for std in self.std:
                noise = torch.normal(
                            mean = 0., 
                            std = std, 
                            size = (chunk_size, latent_sem.shape[1]),
                            device = self.device)
                noises.append(noise)
                log.info(f'Mean norm of noise for {round(std.item(), 7)} std: {noise.norm(dim = 1).mean().item()}')
            noises = torch.cat(noises)
        elif self.distribution == 'uniform_norm':
            log.info(f'Sampling latent sem perturbations from uniform distribution over norm from the interval [0, {self.max_norm}]')
            noise = torch.normal(
                        mean = 0., 
                        std = 1., 
                        size = (self.batch_size, latent_sem.shape[1]),
                        device = self.device)
            noise /= noise.norm(dim = 1).unsqueeze(1)
            scales = torch.rand(size = (self.batch_size, 1), device = self.device)
            scales *= self.max_norm
            noises = noise * scales
            log.info(f'Mean norm of noise: {noises.norm(dim = 1).mean().item()}')
        batch_latent_sem = batch_latent_sem + noises
        return batch_latent_sem

    def make_batch_latent_ddim(self, latent_ddim):
        return latent_ddim.repeat(self.batch_size, 1, 1, 1)
