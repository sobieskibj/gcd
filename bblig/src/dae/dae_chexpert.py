import torch
import torch.nn as nn
from collections import OrderedDict

from .src_chexpert.renderer import render_condition
from .src_chexpert.utils import make_config

class DAECheXpert(nn.Module):
    """
    DiffAE stripped from pl.LightningModule dependency and wrapped
    for inference purposes only.
    """

    def __init__(
            self, 
            config_name: str,
            path_ckpt: str,
            T_encode: int,
            T_render: int):
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
        self.config = make_config(config_name)
        self.sampler_encode = self.config._make_diffusion_conf(T_encode).make_sampler()
        self.sampler_render = self.config._make_diffusion_conf(T_render).make_sampler()
        self.make_model(path_ckpt)
        self.eval()

    def make_model(self, path_ckpt):
        model = self.config.make_model_conf().make_model()
        state_dict = torch.load(path_ckpt, map_location = 'cpu')['state_dict']
        # We only load the ema unet + encoder part of the model
        ema_model_state_dict = OrderedDict(
            [('.'.join(k.split('.')[1:]), v) for k, v in state_dict.items() if 'ema_model' in k])
        model.load_state_dict(ema_model_state_dict)
        model.requires_grad_(False)
        self.model = model

    def forward(self, x):
        # This is just for compatibility with nn.Module and is rather not used
        return self.model(x)

    def rescale(self, x):
        assert x.min() >= 0. and x.max() <= 1.
        return (x - 0.5) * 2

    def get_single_channel(self, x):
        return x[:, 0].unsqueeze(1)

    @torch.no_grad()
    def encode(self, x):
        assert self.config.model_type.has_autoenc()
        x = self.get_single_channel(x) # input must have 1 channel
        x = self.rescale(x)
        return self.model.encoder.forward(x)

    @torch.no_grad()
    def encode_stochastic(self, x, cond, T = None):
        x = self.get_single_channel(x) # input must have 1 channel
        x = self.rescale(x)

        if T is not None:
            sampler = self.config._make_diffusion_conf(T).make_sampler()
        else:
            sampler = self.sampler_encode

        out = sampler.ddim_reverse_sample_loop(
            self.model, x, model_kwargs = {'cond': cond})

        return out['sample']
    
    @torch.no_grad()
    def render(self, noise, cond, T = None):
        if T is not None:
            sampler = self.config._make_diffusion_conf(T).make_sampler()
        else:
            sampler = self.sampler_render

        pred_img = render_condition(
            self.config, self.model, noise, sampler = sampler, cond = cond)
        
        pred_img = (pred_img + 1) / 2
        pred_img = pred_img.repeat(1, 3, 1, 1) # output must have 3 channels
        assert pred_img.shape[1] == 3
        return pred_img