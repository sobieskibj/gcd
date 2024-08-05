import lpips
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import logging
log = logging.getLogger(__name__)

class CounterfactualLossGeneralComponents(nn.Module):
    """
    Class responsible for the calculation of components later used in
    the counterfactual loss. It is disentangled from the actual loss
    since individual components are needed.
    """

    def __init__(
            self, 
            clf,
            lpips_net: str,
            src_img_path: str,
            device: str):
        """
        label_idx - label of interest (for counterfactual explanation)
        """
        super(CounterfactualLossGeneralComponents, self).__init__()
        self.device = torch.device(device)
        self.to(self.device)
        self.classifier = clf.to(self.device)
        self.lpips = lpips.LPIPS(net = lpips_net).to(self.device)
        self.load_src_img(src_img_path, device)
        self.clf_pred_label = self.get_clf_pred_label()

    def forward(self, x):
        assert len(self.src_img.shape) == len(x.shape),\
            f'Invalid shapes: {self.src_img.shape} and {x.shape}'
        with torch.no_grad():
            # NOTE: We need [-1, 1] range
            log.info(f'Minimum: {x.min().item()}')
            log.info(f'Maximum: {x.max().item()}')
            if x.min() >= 0. and x.max() <= 1.:
                log.info('Detected input from [0, 1 range]')
                log.info('Rescaling')
                x = (x - 0.5) * 2
                log.info(f'New minimum: {x.min().item()}')
                log.info(f'New maximum: {x.max().item()}')
            with torch.no_grad():
                batch_lpips = self.lpips(self.src_img, x).flatten()
            batch_predictions = self.classifier(x)
        output = {
            'lpips': batch_lpips.unsqueeze(1),
            'predictions': batch_predictions}
        return output

    @torch.no_grad()
    def get_clf_pred_label(self, img = None):
        # NOTE: We assume that the classifier handles rescaling
        # its input, e.g. ResNet requires [0, 1] range and it
        # scales src_img to this range on its own in self.forward
        if img is None:
            img = self.src_img
            save_src_img_preds = False
        else:
            save_src_img_preds = True

        output = self.classifier(img)

        if self.classifier.task in ['binary_classification_two_outputs', 'multiclass_classification']:
            if self.classifier.use_softmax_and_query_label:
                output_prob = output
            else:
                output_probs = F.softmax(output, dim = 1)
                output_prob = output_probs[:, self.classifier.query_label]
            
            if save_src_img_preds:
                self.src_img_probs = output_probs
                self.src_img_logits = output

        elif self.classifier.task == 'binary_classification_one_output':
            output_prob = F.sigmoid(output, dim = 1)

            if save_src_img_preds:
                self.src_img_probs = output_prob
                self.src_img_logits = output

        elif self.classifier.task == 'multilabel_classification':
            output_probs = F.sigmoid(output)
            output_prob = output_probs[:, self.classifier.query_label]

            if save_src_img_preds:
                self.src_img_probs = output_probs
                self.src_img_logits = output
        
        label = 1 if output_prob > 0.5 else 0
        log.info(f"Class predicted for source image: {label}")
        log.info(f"Probability of positive class: {output_prob.item()}")
        return label
    
    def load_src_img(self, path, device):
        src_img = torchvision.io.read_image(path).unsqueeze(0) / 255
        src_img = (src_img - 0.5) * 2
        self.src_img = src_img.to(self.device)




