import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from .counterfactual_loss_general_components import CounterfactualLossGeneralComponents

import logging
logging.basicConfig(level = logging.INFO)
log = logging.getLogger(__name__)

class CounterfactualLossFromGeneralComponents(nn.Module):
    """
    Class responsible for the calculation of components later used in
    the counterfactual loss. It is disentangled from the actual loss
    since individual components are needed.
    """

    def __init__(
            self, 
            weight_cls: float, 
            weight_lpips: float,
            components: CounterfactualLossGeneralComponents,
            device: str,
            output_dir: str,
            make_output_dir: bool = True,
            init_comps_from_kwargs: bool = False,
            components_kwargs: dict = None):
        super(CounterfactualLossFromGeneralComponents, self).__init__()
        """
        weight_cls - weight assigned by default to classifier component 
        weight_lpips - weight assigned by default to lpips component
        components - CounterfactualLossGeneralComponents object
        """
        self.output_dir = Path(output_dir)
        if make_output_dir:
            self.output_dir.mkdir(parents = True)
        self.device = torch.device(device)
        if init_comps_from_kwargs:
            self.components = CounterfactualLossGeneralComponents(**components_kwargs)
        else:
            self.components = components
        self.weight_cls = weight_cls
        self.weight_lpips = weight_lpips
        self.to(self.device)

    def forward(
            self, 
            x, 
            pos_label_idx: int,
            weight_cls: float = None, 
            weight_lpips: float = None,
            clf_pred_label: int = None):
        assert type(x) == dict
        assert all([k in x.keys() for k in ['lpips', 'predictions']])

        predictions = x['predictions']
        lpips = x['lpips']

        if weight_cls is None and weight_lpips is None:
            weight_cls, weight_lpips = self.weight_cls, self.weight_lpips
        elif weight_cls is not None and weight_lpips is not None:
            weight_cls, weight_lpips = weight_cls, weight_lpips
        else:
            raise ValueError('Both weight_cls and weight_lpips must be provided')
        
        pos_label_logit = predictions[:, pos_label_idx]
        if clf_pred_label is None:
            clf_pred_label = self.components.clf_pred_label
        if clf_pred_label == 0:
            log.info('Classifier predicted negative class for query label, flipping logit sign')
            # Computational graph tracks whether pos_label_logit comes with a positive or negative sign
            # If sign is positive, gradient points to a direction where lpips and logit are minimized
            #     -> class changes from positive to negative
            # If sign is negative, gradient points to a direction where lpips is minimized and logit
            # is maximized. Because we always track the logit for positive class, maximizing this logit
            # flips the label from negative to positive
            #     -> class changes from negative to positive
            pos_label_logit = - pos_label_logit
        else:
            log.info('Classifier predicted positive class for query label, leaving logit sign as is')

        if pos_label_logit.shape != lpips.shape:
            lpips = lpips.reshape(pos_label_logit.shape)

        log.info('Calculating standard counterfactual loss')
        loss = weight_cls * pos_label_logit + weight_lpips * lpips
        return loss
    
    @torch.no_grad()
    def get_components(self, x):
        return self.components(x)

    def get_query_label_probability(self, x, from_logits = True):
        return self.get_label_probability(x, self.components.classifier.query_label, from_logits)

    def get_query_label_logit(self, logits):
        return self.get_label_logit(logits, self.components.classifier.query_label)
    
    def get_label_probability(self, x, label_idx, from_logits = True):
        # We do not have the probabilities at hand so we transform the logits based
        # on classifier's task
        if from_logits:
            if self.components.classifier.task in ['binary_classification_two_outputs', 'multiclass_classification']:
                x = F.softmax(x, dim = 1)

            elif self.components.classifier.task == 'binary_classification_one_output':
                x = F.sigmoid(x, dim = 1)

            elif self.components.classifier.task == 'multilabel_classification':
                x = F.sigmoid(x)
                
        return x[:, label_idx]
    
    def get_label_logit(self, x, label_idx):
        return x[:, label_idx]
    