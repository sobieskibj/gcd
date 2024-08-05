import random
import wandb
import hydra
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from losses import GeneralMulticomponentProxyLoss
from utils import log_wandb_scatter

import logging
log = logging.getLogger(__name__)

activations = {
    'relu': nn.ReLU(),
    'sigmoid': nn.Sigmoid(),
    'silu': nn.SiLU(),
    'elu': nn.ELU(),
    'hardswish': nn.Hardswish(),
    'logsigmoid': nn.LogSigmoid(),
    'selu': nn.SELU(),
    'celu': nn.CELU(),
    'gelu': nn.GELU(),
    'mish': nn.Mish(),
    'softplus': nn.Softplus(),
    'softsign': nn.Softsign(),
    'tanh': nn.Tanh(),
    'tanshrink': nn.Tanhshrink()}

class GeneralMulticomponentMLP(nn.Module):
    """
    Simple MLP with nonlinear activations.

    It outputs a 'predictions' vector which correspond to classifier's predictions
    and a 'lpips' scalar which corresponds to the LPIPS value between source image
    and target image.
    """

    def __init__(
            self, 
            shapes: list,
            batch_size: int,
            n_val_batches: int,
            val_loss_history_length: int,
            loss_kwargs: dict,
            optimizer,
            device: str,
            output_dir: str,
            activ_fn: str = 'relu',
            use_batch_norm: bool = False,
            make_output_dir: bool = True,
            init_optimizer: bool = True,
            save: bool = True,
            log_wandb: bool = True):
        """
        shapes - list of layers' shapes. Last element must be 
            equal to the number of classifier's outputs plus one
            for LPIPS estimation.
        """
        super(GeneralMulticomponentMLP, self).__init__()
        
        self.output_dir = Path(output_dir)
        if make_output_dir:
            self.output_dir.mkdir(parents = True)
        self.device = torch.device(device)
        self.save = save
        self.log_wandb = log_wandb
        self.shapes = shapes
        self.activ_fn = activ_fn
        self.use_batch_norm = use_batch_norm
        self.batch_size = batch_size
        self.n_val_batches = n_val_batches
        self.val_loss_history_length = val_loss_history_length
        self.training_data = []
        self.validation_data = []
        self.data_save_counter = -1
        self.continue_training = True
        self.val_loss_history = []

        self.init_optimizer = init_optimizer
        self.optimizer_partial = optimizer
        self.model = self.make_model()
        self.loss = GeneralMulticomponentProxyLoss(**loss_kwargs)
        self.to(self.device)

    def make_model(self):
        layers = []
        for i in range(len(self.shapes) - 1):
            layers.append(nn.Linear(self.shapes[i], self.shapes[i + 1]))
            if not i + 1 == len(self.shapes) - 1:
                if self.use_batch_norm:
                    layers.append(nn.BatchNorm1d(self.shapes[i + 1]))
                layers.append(activations[self.activ_fn])
        model = nn.Sequential(*layers).to(self.device)
        if self.init_optimizer:
            self.optimizer = self.optimizer_partial(params = model.parameters())
        return model
    
    def forward(self, x):
        x = self.model(x)
        predictions = x[:, :-1] # logits
        lpips = F.sigmoid(x[:, -1]).unsqueeze(1) # [0, 1] range, column vector
        output = {
            'lpips': lpips,
            'predictions': predictions}
        return output

    def run_epoch(self, data, epoch_idx, iter_idx):
        if self.continue_training:
            assert all([k in ['latent_sem', 'components'] for k in data.keys()])
            if epoch_idx == 0:
                self.prepare_train_val_data(data)
            self.shuffle_training_data()

            # Training loop
            self.train()
            log_freq = int(0.2 * len(self.training_data))
            for batch_idx, batch in enumerate(self.training_data):
                batch_inputs, batch_targets = batch
                self.optimizer.zero_grad()
                batch_outputs = self(batch_inputs)
                loss = self.loss(batch_outputs, batch_targets)
                loss.backward()
                self.optimizer.step()
                if batch_idx % log_freq == 0:
                    log.info(f'Batch: {batch_idx}, Loss: {loss.item()}')
                if self.log_wandb:
                    wandb.log({f'proxy/loss/iter:{iter_idx}/train': loss.item()})

            # Validation loop
            self.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_idx, batch in enumerate(self.validation_data):
                        batch_inputs, batch_targets = batch
                        batch_outputs = self(batch_inputs)
                        loss = self.loss(batch_outputs, batch_targets)
                        val_loss += loss.item()

            # Early stopping
            mean_val_loss = val_loss / len(self.validation_data)
            if self.log_wandb:
                wandb.log({f'proxy/loss/iter:{iter_idx}/val': mean_val_loss})
            log.info(f'Validation loss: {mean_val_loss}')
            self.val_loss_history = self.val_loss_history[-self.val_loss_history_length:]
            if len(self.val_loss_history) > 0:
                running_mean = sum(self.val_loss_history) / len(self.val_loss_history)
                log.info(f'Validation running average: {running_mean}')
                if mean_val_loss > running_mean:
                    log.info('Applying early stopping')
                    self.continue_training = False
            self.val_loss_history.append(mean_val_loss)
        else:
            log.info('Validation loss not improving, training discontinued')

    def prepare_train_val_data(self, data):
        log.info('Preparing training and validation data')
        output_dir = self.output_dir / f'{self.data_save_counter}'
        output_dir.mkdir(exist_ok = True)
        permutation = torch.randperm(len(data['latent_sem']) * data['latent_sem'][0].shape[0])
        latent_sem_batches = self.make_latent_sem_batches(data['latent_sem'], output_dir, permutation)
        comps_batches = self.make_comps_batches(data['components'], output_dir, permutation)
        batches = [(e1, e2) for e1, e2 in zip(latent_sem_batches, comps_batches)]
        random.shuffle(batches)
        self.validation_data += batches[:self.n_val_batches]
        self.training_data += batches[self.n_val_batches:]
        log.info(f'Number of training batches: {len(self.training_data)}')
        log.info(f'Number of validation batches: {len(self.validation_data)}')

    def shuffle_training_data(self):
        random.shuffle(self.training_data)

    def make_latent_sem_batches(self, latent_sems, save_dir, permutation):
        latent_sems_all = torch.cat(latent_sems)[permutation]
        if self.save:
            log.info(f'Saving latent sems from proxy dataset to {save_dir}')
            torch.save(latent_sems_all, save_dir / 'training_latent_sem.pt')
        return torch.split(latent_sems_all, self.batch_size)

    def make_comps_batches(self, comps, save_dir, permutation):
        lpips_all = torch.cat([e['lpips'] for e in comps])
        predictions_all = torch.cat([e['predictions'] for e in comps])
        if len(predictions_all.shape) == 1:
            predictions_all = predictions_all.unsqueeze(1)
        comps_all = torch.hstack([lpips_all, predictions_all])[permutation]
        if self.save:
            log.info(f'Saving components from proxy dataset to {save_dir}')
            torch.save(comps_all, save_dir / 'training_components.pt')
        comps_batches = torch.split(comps_all, self.batch_size)
        return [{'lpips': batch[:, 0].unsqueeze(1), 'predictions': batch[:, 1:]} for batch in comps_batches]

    def line_search_validation(
            self, 
            batch_latent_sem, 
            batch_components, 
            query_label, 
            iter, 
            step_sizes, 
            weights_pairs, 
            ce_ids, 
            hess_vec_ids):
        # Save line search results
        output_dir = self.output_dir / f'{self.data_save_counter}'
        log.info(f'Saving line search batch to {output_dir}')
        batch_outputs = self(batch_latent_sem)
        torch.save(batch_outputs, output_dir / 'line_search_predicted_components.pt')
        torch.save(batch_components, output_dir / 'line_search_components.pt')
        torch.save(batch_latent_sem, output_dir / 'line_search_latent_sems.pt')
        loss = self.loss(batch_outputs, batch_components)
        log.info(f'Proxy loss on line search: {loss.item()}')

        # Log info file
        log.info(f'Saving info file to {output_dir}')
        df_data = np.vstack([
            step_sizes.numpy(force = True).flatten(),
            batch_latent_sem.norm(dim = 1).numpy(force = True).flatten(),
            hess_vec_ids,
            np.array([ws[0] for ws in weights_pairs]),
            np.array([ws[1] for ws in weights_pairs]),
            batch_outputs['predictions'][:, query_label].numpy(force = True).flatten(),
            batch_components['predictions'][:, query_label].numpy(force = True).flatten(),
            batch_outputs['lpips'].numpy(force = True).flatten(),
            batch_components['lpips'].numpy(force = True).flatten(),
            ce_ids.numpy(force = True)],).T
        df = pd.DataFrame(
            df_data, 
            columns = ['step_size', 'norm', 'from_hess_vec', 'weight_cls', 'weight_lps', 'pred_clf_logit', 'clf_logit', 'pred_lpips', 'lpips', 'is_ce'])
        df.to_csv(output_dir / 'info.csv')

        # Log scatters to wandb showing proxy error wrt to step size
        log_wandb_scatter(
            batch_outputs['lpips'].flatten().numpy(force = True), 
            batch_components['lpips'].flatten().numpy(force = True), 
            'proxy predicted lpips', 'ground truth lpips',
            f'proxy/line_search/lpips/iter:{iter}')
        log_wandb_scatter(
            batch_outputs['predictions'][:, query_label].flatten().numpy(force = True), 
            batch_components['predictions'][:, query_label].flatten().numpy(force = True), 
            'proxy predicted positive logit', 'ground truth positive logit',
            f'proxy/line_search/logit/iter:{iter}')

    def increment_data_save_counter(self):
        # We increment the counter to track the epoch number
        self.data_save_counter += 1

    def reinit(self, save = False):
        if save:
            output_dir = self.output_dir / f'{self.data_save_counter}'
            log.info(f'Saving model to {output_dir}')
            torch.save(self.model, output_dir / 'model.pt')
        self.model = self.make_model()
        if self.log_wandb:
            wandb.watch(self.model)
        self.val_loss_history = []
        self.continue_training = True
        
    def load_weights_from_ckpt(self, ckpt_path):
        log.info('Loading weights')
        self.model = torch.load(ckpt_path, map_location = self.device)

