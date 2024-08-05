import torch
import torch.nn as nn
import torch.nn.functional as F

class GeneralMulticomponentProxyLoss(nn.Module):
    """
    Class responsible for the calculation of the multicomponent proxy loss.
    """

    def __init__(
            self, 
            weight_cls: float, 
            weight_lpips: float):
        """
        weight_cls - weight of the probability component
        weight_lpips - weight of the LPIPS component
        """
        super(GeneralMulticomponentProxyLoss, self).__init__()
        self.weight_cls = weight_cls
        self.weight_lpips = weight_lpips

    def forward(self, inputs, targets, labels: list = None):
        assert type(inputs) == dict, f'Wrong type: {type(inputs)}'
        assert type(targets) == dict, f'Wrong type: {type(targets)}'
        assert inputs.keys() == targets.keys(), "Keys of inputs and targets don't match"

        pred_preds = inputs['predictions']
        target_preds = targets['predictions']

        pred_lpips = inputs['lpips']
        target_lpips = targets['lpips']

        assert pred_preds.shape == target_preds.shape
        assert pred_lpips.shape == target_lpips.shape

        loss = self.weight_cls * F.mse_loss(pred_preds, target_preds) + \
            self.weight_lpips * F.mse_loss(pred_lpips, target_lpips)
        return loss