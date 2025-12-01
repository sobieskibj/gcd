import lpips
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from datasets import load_dataset
from PIL import Image
import os

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
            src_img_path: str = None,
            device: str = 'cuda:0',
            # Optional: load source image from HF dataset by label if src_img_path is None
            hf_dataset_name: str = None,    # e.g., 'imagenet-1k'
            hf_split: str = 'train',
            hf_label: int = None,
            hf_index: int = 0,
            image_size: int = 256):
        """
        label_idx - label of interest (for counterfactual explanation)
        """
        super(CounterfactualLossGeneralComponents, self).__init__()
        self.device = torch.device(device)
        self.to(self.device)
        self.classifier = clf.to(self.device)
        self.lpips = lpips.LPIPS(net = lpips_net).to(self.device)
        # Load source image: prefer local path, otherwise try HF dataset
        if src_img_path is not None:
            self.load_src_img(src_img_path, device)
        else:
            # Fallback to HF if params provided or env vars present
            ds_name = hf_dataset_name or os.getenv('HF_DATASET_NAME', None)
            ds_split = hf_split
            ds_label = hf_label if hf_label is not None else os.getenv('HF_LABEL', None)
            ds_index = hf_index if hf_index is not None else int(os.getenv('HF_INDEX', 0))
            ds_token = os.getenv('HF_TOKEN', None)
            ds_cache = os.getenv('DATASET_CACHE', None)
            if ds_name is None or ds_label is None:
                raise ValueError("CounterfactualLossGeneralComponents: Provide src_img_path or HF dataset parameters (hf_dataset_name & hf_label).")
            ds_label = int(ds_label)
            self.load_src_img_from_hf(
                dataset_name = ds_name,
                split = ds_split,
                label = ds_label,
                index = ds_index,
                hf_token = ds_token,
                cache_dir = ds_cache,
                image_size = image_size,
            )
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
                # For multiclass, threshold 0.5 is often too strict (1k classes).
                # Use argmax to decide if query_label is the predicted class.
                if self.classifier.task == 'multiclass_classification':
                    pred_idx = output_probs.argmax(dim = 1)
                    label = (pred_idx == self.classifier.query_label).long().item()
                    if save_src_img_preds:
                        self.src_img_probs = output_probs
                        self.src_img_logits = output
                    log.info(f"Argmax predicted class: {pred_idx.item()}, query_label: {self.classifier.query_label}")
                    log.info(f"Probability of query class: {output_prob.item()}")
                    return label

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

    @torch.no_grad()
    def load_src_img_from_hf(
        self,
        dataset_name: str,
        split: str,
        label: int,
        index: int = 0,
        hf_token: str = None,
        cache_dir: str = None,
        image_size: int = 256,
    ):
        log.info(f'Loading source image from HF dataset: {dataset_name}, split: {split}, label: {label}, index: {index}')
        # Use streaming to avoid slow filter materialization
        stream_kwargs = {"streaming": True}
        if hf_token is not None:
            stream_kwargs["token"] = hf_token
        if cache_dir is not None:
            stream_kwargs["cache_dir"] = cache_dir
        ds_stream = load_dataset(dataset_name, split=split, **stream_kwargs)

        found = -1
        sample = None
        for ex in ds_stream:
            if ex.get("label", None) == label:
                found += 1
                if found == index:
                    sample = ex
                    break
        if sample is None:
            raise ValueError(f"Could not find occurrence index {index} for label {label} in {dataset_name}/{split}")
        img = sample['image']
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        transform = T.Compose([
            T.CenterCrop(image_size),
            T.Resize(image_size),
            T.ToTensor(),          # [0,1]
        ])
        src_img = transform(img).unsqueeze(0)
        # Scale to [-1,1] for loss computations (classifier handles rescale internally if needed)
        src_img = (src_img - 0.5) * 2
        self.src_img = src_img.to(self.device)




