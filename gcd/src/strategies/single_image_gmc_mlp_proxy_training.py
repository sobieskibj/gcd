import torch
import torchvision
import torch.nn.functional as F
import pandas as pd
from pathlib import Path
from typing import List
from collections import OrderedDict

from dae import DAE, DAECheXpert
from proxies import GeneralMulticomponentMLP
from losses import CounterfactualLossFromGeneralComponents
from strategies import Strategy
from utils import *

import logging
logging.basicConfig(level = logging.INFO)
log = logging.getLogger(__name__)

class SingleImageGMCMLPProxyTraining(Strategy):
    """
    Single-Image General MultiComponent MLP proxy training

    Trains a multicomponent MLP proxy on a single image in a loop:
    1. Generate variations of an image
    2. Calculate counterfactual loss components
    3. Store (latent_sem, components) pairs in a general buffer
    3. Train MLP to approximate these components using the general buffer
    4. Take gradient of counterfactual loss wrt MLP input (latent rep)
    5. Perform line search in gradient and hessian eigenvectors directions
    6. Reinitialize MLP to train in the next iteration with bigger dataset
    """

    def __init__(
        self,
        src_img_path: str,
        n_iters: int,
        n_steps_dae: int,
        n_steps_proxy: int,
        min_max_step_size: list,
        center_to_src_latent_sem: bool,
        line_search_weight_cls_list: list,
        line_search_weight_lpips_list: list,
        n_hessian_eigenvecs: int,
        device: str,
        output_dir: str,
        mc_mlp_proxy_kwargs: dict, 
        dae_kwargs: dict,
        ce_loss_kwargs: dict,
        dae_type: str = 'default'):
        """
        src_img_path - path of the source image
        n_iters - number of outer loop iterations
        n_steps_dae - number of iterations of the inner DAE loop in each outer iteration
            Note: For each outer iteration, n_steps_dae * self.dae.batch_size is the effective
            number of synthetic images generated before proceeding to inner proxy loop
        n_steps_proxy - number of epochs for proxy training in each outer iteration
            Note: the size of the  dataset for proxy training is n_steps_dae * self.dae.batch_size * n_iters
        min_max_step_size - coefficients that determine the  min and max size of the gradient step,
            intermediate steps are distributed uniformly between these values and total number of 
            step sizes is equal to self.dae.batch_size
        center_to_src_latent_sem - whether to center all latent sems at src_latent_sem
        n_hessian_eigenvecs - number of hessian eigenvectors used in line search
        """
        self.n_iters = n_iters
        self.n_steps_dae = n_steps_dae
        self.n_steps_proxy = n_steps_proxy
        self.n_hessian_eigenvecs = n_hessian_eigenvecs
        self.min_max_step_size = min_max_step_size
        self.center_to_src_latent_sem = center_to_src_latent_sem
        self.weight_cls_lpips_pairs = [(w_cls, w_lpips) for w_cls, w_lpips \
                                       in zip(line_search_weight_cls_list, line_search_weight_lpips_list)]

        self.device = torch.device(device)
        self.device_cpu = torch.device('cpu')
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents = True)
        
        self.dae = self.get_dae_class(dae_type)(**dae_kwargs)
        self.proxy = GeneralMulticomponentMLP(**mc_mlp_proxy_kwargs)
        self.ce_loss = CounterfactualLossFromGeneralComponents(**ce_loss_kwargs)

        self.load_src_img(src_img_path)
        self.clear_proxy_training_data()

    def get_dae_class(self, dae_type):
        if dae_type == 'default':
            return DAE
        elif dae_type == 'chexpert':
            return DAECheXpert

    def load_src_img(self, path):
        log.info(f'Loading source image and saving to {self.output_dir}')
        src_img = torchvision.io.read_image(path).unsqueeze(0) / 255
        torchvision.utils.save_image(src_img, self.output_dir / 'src_img.png')
        # Scale to [-1, 1] for DAE
        src_img = (src_img - 0.5) * 2
        self.src_img = src_img.to(self.device)

    def get_src_img(self, scale = True):
        if scale:
            return (self.src_img + 1) / 2
        else:
            return self.src_img

    def pre_main_loop(self):
        # Get latent reps of src image
        log.info('Encoding source image')
        self.src_latent_sem = self.dae.encode(self.src_img).to(self.device)
        self.src_latent_ddim = self.dae.encode_stochastic(
            self.src_img, self.src_latent_sem).to(self.device)
        log.info(f'Saving latent representations to {self.output_dir}')
        torch.save(self.src_latent_sem, self.output_dir / 'src_latent_sem.pt')
        torch.save(self.src_latent_ddim, self.output_dir / 'src_latent_ddim.pt')

    def pre_dae_loop(self, iter):
        pass

    def dae_step(self, step_idx, iter):
        # Make dir for step outputs
        output_dir = self.output_dir / f'iter:{iter}' / f'step:{step_idx}'
        output_dir.mkdir(parents = True)

        # Make latent reps
        log.info(f'Step: {step_idx}')
        log.info('Making batches of perturbed latent representations')
        batch_latent_sem = self.dae.make_batch_latent_sem(self.src_latent_sem)
        batch_latent_ddim = self.dae.make_batch_latent_ddim(self.src_latent_ddim)
        log.info(f'Saving perturbed latent sems to {output_dir}')
        torch.save(batch_latent_sem, output_dir / 'latent_sems.pt')

        # Generate batch of new synthetic images
        log.info('Rendering synthetic images')
        batch_imgs = self.dae.render(batch_latent_ddim, batch_latent_sem)
        log.info(f'Saving synthetic images to {output_dir}')
        torchvision.utils.save_image(batch_imgs, output_dir / 'imgs.png')
        # NOTE: DAE requires input to be in [-1, 1] but outputs imgs in [0, 1]
        
        # Compute counterfactual loss components
        log.info('Calculating counterfactual loss components')
        # In general case, we use counterfactual loss to provide us with
        # the probability assigned to the class of interest. It handles multiple scenarios:
        # - binary classification
        # - multiclass classification
        # - multilabel classification
        batch_components = self.ce_loss.get_components(batch_imgs)
        batch_lpips = batch_components['lpips']
        batch_predictions = batch_components['predictions']
        batch_probs = self.ce_loss.get_query_label_probability(batch_predictions)
        batch_logits = self.ce_loss.get_query_label_logit(batch_predictions)
        
        # Log pareto fronts
        log_wandb_scatter(
            batch_logits.flatten().numpy(force = True), 
            batch_lpips.flatten().numpy(force = True), 
            'logits from classifier', 'lpips', 
            f'dae/pareto/logits/iter:{iter}/{step_idx}')
        log_wandb_scatter(
            batch_probs.flatten().numpy(force = True), 
            batch_lpips.flatten().numpy(force = True), 
            'probabilities from classifier', 'lpips', 
            f'dae/pareto/probs/iter:{iter}/{step_idx}')

        # Log info 
        ce_ids = batch_probs < 0.5 if self.ce_loss.components.clf_pred_label == 1 else batch_probs > 0.5
        log.info(f'Number of CEs in synthetic images: {batch_probs[ce_ids].shape[0]}')
        log.info(f'Saving info file to {output_dir}')
        df_data = np.vstack([
            (batch_latent_sem.clone() - self.src_latent_sem).norm(dim = 1).numpy(force = True).flatten(),
            batch_probs.numpy(force = True),
            batch_lpips.numpy(force = True).flatten(),
            ce_ids.numpy(force = True)]).T
        df = pd.DataFrame(
            df_data, 
            columns = ['norm', 'clf_prob', 'lpips', 'is_ce'])
        df.to_csv(output_dir / 'info.csv')

        # Log diffs
        log.info(f'Saving diffs to {output_dir}')
        src_img = self.get_src_img()
        diffs_all = (batch_imgs - src_img).abs()
        diffs_all = diffs_all / diffs_all.amax(dim = (1, 2, 3)).view(-1, 1, 1, 1)
        diffs_all_grid = torchvision.utils.make_grid(diffs_all)
        torchvision.utils.save_image(diffs_all_grid, output_dir / 'diffs_all.png')

        # Log counterfactuals if any
        if torch.any(ce_ids):
            grid_ce = batch_imgs.clone()
            grid_ce[~ce_ids] = 0
            grid_ce = torchvision.utils.make_grid(grid_ce)
            wandb.log({f'dae/counterfactuals/iter:{iter}/step:{step_idx}': wandb.Image(grid_ce)})
            log.info(f'Saving counterfactuals to {output_dir}')
            torchvision.utils.save_image(grid_ce, output_dir / 'counterfactuals.png')
            log.info(f'Saving counterfactuals diffs to {output_dir}')
            diffs_ces = diffs_all.clone()
            diffs_ces[~ce_ids] = 0
            diffs_ces_grid = torchvision.utils.make_grid(diffs_ces)
            torchvision.utils.save_image(diffs_ces_grid, output_dir / 'diffs_ces.png')

        # Save proxy training data
        log.info('Saving data for proxy training')            
        if self.center_to_src_latent_sem:
            batch_latent_sem = batch_latent_sem - self.src_latent_sem
        self.save_proxy_training_data(batch_latent_sem, batch_components)

    def post_dae_loop(self, iter):
        pass

    def pre_proxy_loop(self, iter):
        pass

    def proxy_step(self, step_idx, iter):
        log.info(f'Step: {step_idx}')
        self.proxy.run_epoch(self.proxy_training_data, step_idx, iter)
            
    def post_proxy_loop(self, iter):
        # Take steps of different magnitude in negative gradient direction aka line search
        # Gradient is calculated for one or more (weight_cls, weight_lpips) pairs and
        # ce loss can take into account additional label for which the probability should change
        grads_dict = self.get_grads_dict()
        chunk_size = self.dae.batch_size // len(grads_dict)
        n_directions = len(self.weight_cls_lpips_pairs) + self.n_hessian_eigenvecs
        step_sizes_stack = torch.linspace(
            *self.min_max_step_size, 
            chunk_size,
            device = self.device).repeat(n_directions).unsqueeze(1)
        grads_stack = torch.stack([*grads_dict.values()]).repeat_interleave(chunk_size, 0).squeeze()
        batch_latent_sem = self.src_latent_sem - step_sizes_stack * grads_stack
        batch_latent_ddim = self.dae.make_batch_latent_ddim(self.src_latent_ddim)

        # Generate images from line search latent sems
        log.info('Rendering line search images')
        batch_imgs = self.dae.render(batch_latent_ddim, batch_latent_sem)
        output_dir = self.output_dir / f'iter:{iter}'
        log.info(f'Saving images from line search to {output_dir}')
        torchvision.utils.save_image(batch_imgs, output_dir / 'line_search_imgs.png')

        # Get components
        batch_components = self.ce_loss.get_components(batch_imgs)
        batch_predictions = batch_components['predictions']
        batch_probs = self.ce_loss.get_query_label_probability(batch_predictions)
        batch_logits = self.ce_loss.get_query_label_logit(batch_predictions)
        batch_lpips = batch_components['lpips']
        ce_ids = batch_probs < 0.5 if self.ce_loss.components.clf_pred_label == 1 else batch_probs > 0.5
        hess_vec_ids = [0] * chunk_size * len(self.weight_cls_lpips_pairs) + [1] * chunk_size * self.n_hessian_eigenvecs

        # Get components and run validation on proxy
        if self.center_to_src_latent_sem:
            batch_latent_sem = batch_latent_sem - self.src_latent_sem

        weights_pairs = [pair for pair in self.weight_cls_lpips_pairs for _ in range(chunk_size)]
        if self.n_hessian_eigenvecs > 0:
            weights_pairs += [(1.0, 0.0) for _ in range(chunk_size * self.n_hessian_eigenvecs)]

        self.proxy.line_search_validation(
            batch_latent_sem = batch_latent_sem, 
            batch_components = batch_components, 
            query_label = self.ce_loss.components.classifier.query_label, 
            iter = iter,
            step_sizes = step_sizes_stack,
            weights_pairs = weights_pairs,
            ce_ids = ce_ids,
            hess_vec_ids = hess_vec_ids)

        # Calculate counterfactual loss on line search images
        batch_losses = self.ce_loss(
                batch_components, 
                pos_label_idx = self.ce_loss.components.classifier.query_label)

        # Log info about line search
        log.info(f'Number of CEs in synthetic images: {batch_probs[ce_ids].shape[0]}')
        batch_losses_argmin = torch.argmin(batch_losses)
        batch_losses_argmax = torch.argmax(batch_losses)
        log.info(f'Line search lowest loss: {batch_losses[batch_losses_argmin].item()}')
        log.info(f'Step size for image with lowest loss: {step_sizes_stack[batch_losses_argmin].item()}')
        log.info(f'Lowest loss achieved for (weight_cls, weight_lpips) pair: {weights_pairs[batch_losses_argmin]}')
        log.info(f'Line search highest loss: {batch_losses[batch_losses_argmax].item()}')
        log.info(f'Step size for image with highest loss: {step_sizes_stack[batch_losses_argmax].item()}')
        log.info(f'Highest loss achieved for (weight_cls, weight_lpips) pair: {weights_pairs[batch_losses_argmax]}')

        # Log pareto fronts
        log_wandb_scatter(
            batch_logits.flatten().numpy(force = True), 
            batch_lpips.flatten().numpy(force = True), 
            'logits from classifier', 'lpips', 
            f'proxy/line_search/pareto/logits/iter:{iter}')
        log_wandb_scatter(
            batch_probs.flatten().numpy(force = True), 
            batch_lpips.flatten().numpy(force = True), 
            'probabilities from classifier', 'lpips', 
            f'proxy/line_search/pareto/probs/iter:{iter}')

        # Log info and counterfactuals if any
        ce_ids = batch_probs < 0.5 if self.ce_loss.components.clf_pred_label == 1 else batch_probs > 0.5
        log.info(f'Number of CEs in line search batch: {batch_probs[ce_ids].shape[0]}')

        log.info(f'Saving info file to {output_dir}')
        df_data = np.vstack([
            step_sizes_stack.numpy(force = True).flatten(),
            hess_vec_ids,
            np.array([ws[0] for ws in weights_pairs]),
            np.array([ws[1] for ws in weights_pairs]),
            batch_probs.numpy(force = True),
            batch_lpips.numpy(force = True).flatten(),
            ce_ids.numpy(force = True)]).T
        df = pd.DataFrame(
            df_data, 
            columns = ['step_size', 'from_hess_vec', 'weight_cls', 'weight_lps', 'clf_prob', 'lpips', 'is_ce'])
        df.to_csv(output_dir / 'info.csv')
        
        src_img = self.get_src_img()
        diffs = (batch_imgs - src_img).abs()
        diffs_scaled = diffs / diffs.amax(dim = (1, 2, 3)).view(-1, 1, 1, 1)
        torchvision.utils.save_image(diffs_scaled, output_dir / 'diffs_all.png')
        if torch.any(ce_ids):
            log.info(f'Saving counterfactuals to {output_dir}')
            ce_s = batch_imgs.clone()
            ce_s[~ce_ids] = 0.
            diffs_ce_s = diffs_scaled.clone()
            diffs_ce_s[~ce_ids] = 0.
            torchvision.utils.save_image(ce_s, output_dir / 'ce_s.png')
            torchvision.utils.save_image(diffs_ce_s, output_dir / 'diffs_ce_s.png')
            grid_ce_s = torchvision.utils.make_grid(ce_s)
            wandb.log({f'line_search/counterfactuals/iter:{iter}': wandb.Image(grid_ce_s)})

        # Reset proxy parameters and training data
        self.proxy.reinit(save = True)
        self.clear_proxy_training_data()

    def post_main_loop(self):
        pass

    def get_grads_dict(self):
        if self.center_to_src_latent_sem:
            input = torch.zeros_like(self.src_latent_sem)
        else:
            input = self.src_latent_sem

        grads_dict = OrderedDict()

        for dir_idx, (weight_cls, weight_lpips) in enumerate(self.weight_cls_lpips_pairs):
            log.info(f'Computing gradient for weight_cls: {weight_cls} and weight_lpips: {weight_lpips}')
            input.requires_grad_()
            outputs = self.ce_loss(
                self.proxy(input),
                pos_label_idx = self.ce_loss.components.classifier.query_label,
                weight_cls = weight_cls,
                weight_lpips = weight_lpips)
            grad = torch.autograd.grad(outputs, input)[0]
            input.requires_grad_(False)
            output_dir = self.proxy.output_dir / f'{self.proxy.data_save_counter}'
            orig_grad_norm = grad.norm().item()
            log.info(f'Original gradient norm: {orig_grad_norm}')
            log.info(f'Saving gradient to {output_dir}')
            grad_idx = f'w-cls:{weight_cls}_w-lps:{weight_lpips}_norm:{orig_grad_norm}'
            grad = F.normalize(grad)
            torch.save(grad, output_dir / f'{dir_idx}_grad_{grad_idx}.pt')
            grads_dict[grad_idx] = grad.squeeze()

        if self.n_hessian_eigenvecs > 0:
            n_pairs = len(self.weight_cls_lpips_pairs)
            log.info(f'Computing hessian for weight_cls: 1.0 and weight_lpips: 0.0')
            input.requires_grad_()
            func = lambda x: self.ce_loss(
                self.proxy(x), 
                pos_label_idx = self.ce_loss.components.classifier.query_label,
                weight_cls = 1.0,
                weight_lpips = 0.0)
            hessian = torch.autograd.functional.hessian(func, input)
            hessian = hessian.squeeze()
            input.requires_grad_(False)
            log.info(f'Saving hessian to {output_dir}')
            hessian_abs = hessian.abs()
            h_min = hessian_abs.min()
            h_max = hessian_abs.max()
            torchvision.utils.save_image((hessian - h_min) / (h_max - h_min), output_dir / 'hessian_abs.png')
            torch.save(hessian, output_dir / 'hessian.pt')
            log.info('Computing hessian eigendecomposition')
            eig_vals, eig_vecs = torch.linalg.eig(hessian)
            n_real_eig_vals = eig_vals.isreal().sum().item()
            log.info(f'Number of real eigenvalues: {n_real_eig_vals}')
            log.info('Check that we are properly sorting')
            _, permutation = eig_vals.real.abs().sort(descending = True)
            eig_vals, eig_vecs = eig_vals[permutation], eig_vecs.T[permutation]
            log.info(f'Highest eigenvalue: {eig_vals[0].item()}')
            one = torch.ones(1, device = eig_vecs.device)
            for eigenvec_idx, (eigenval, eigenvec) in enumerate(zip(eig_vals, eig_vecs)):
                assert torch.isclose(eigenvec.norm(), one)
                if eigenvec_idx == self.n_hessian_eigenvecs:
                    break
                log.info(f'Saving hessian eigenvec number {eigenvec_idx} corresponding to eigenvalue: {eigenval.item()}')
                vec_idx = f'eigval-real:{eigenval.real.item()}_eigval-imag:{eigenval.imag.item()}_w-cls:1.0_w-lps:0.0'
                torch.save(eigenvec.real, output_dir / f'{eigenvec_idx + n_pairs}_hess_eigv-{vec_idx}.pt')
                if eigenval.real > 0.:
                    # logits increase when positive
                    eigenvec = - eigenvec
                grads_dict[vec_idx] = eigenvec.real
        else:
            log.info('Skipping hessian computation')
        return grads_dict

    def save_proxy_training_data(self, batch_latent_sem, batch_components):
        self.proxy_training_data['latent_sem'].append(batch_latent_sem)
        self.proxy_training_data['components'].append(batch_components)
        
    def clear_proxy_training_data(self):
        self.proxy.increment_data_save_counter()
        self.proxy_training_data = {
            'latent_sem': [],
            'components': []}
        