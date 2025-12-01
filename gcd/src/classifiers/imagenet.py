import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as tt

import logging
logging.basicConfig(level = logging.INFO)
log = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

class ImageNetResNet(torch.nn.Module):
    def __init__(
            self,
            img_size: int = 256,
            query_label: int = 340,
            use_softmax_and_query_label: bool = False,
            task: str = 'multiclass_classification',
            model_name: str = 'resnet50',
            weights: str = 'IMAGENET1K_V2',
            path_to_weights: str = None,
            debug_save_path: str = None):
        super().__init__()
        self.img_size = img_size
        self.task = task
        self.query_label = int(query_label)
        self.use_softmax_and_query_label = use_softmax_and_query_label
        self.transforms = tt.Compose([
            tt.Normalize(mean = IMAGENET_MEAN, std = IMAGENET_STD)
        ])
        self.debug_save_path = debug_save_path
        self.model = self._build_model(model_name, weights, path_to_weights)
        self.model.eval()

    def _build_model(self, model_name, weights, path_to_weights):
        log.info(f'Building torchvision model: {model_name}')
        if hasattr(torchvision.models, model_name):
            ctor = getattr(torchvision.models, model_name)
        else:
            raise ValueError(f'Unknown model_name: {model_name}')
        if path_to_weights is None:
            # Load torchvision pretrained weights
            try:
                w_enum = getattr(torchvision.models, f'{model_name.upper()}_Weights')
                w = getattr(w_enum, weights)
                model = ctor(weights = w)
            except Exception:
                log.warning('Falling back to default pretrained weights')
                model = ctor(weights = 'IMAGENET1K_V2')
        else:
            model = ctor(weights = None)
            log.info(f'Loading classifier weights from: {path_to_weights}')
            ckpt = torch.load(path_to_weights, map_location = 'cpu')
            state_dict = ckpt.get('state_dict', ckpt)
            # if state_dict keys are prefixed (e.g., 'model.'), try to strip common prefixes
            if any(k.startswith('model.') for k in state_dict.keys()):
                state_dict = {k.split('model.', 1)[1]: v for k, v in state_dict.items() if k.startswith('model.')}
            missing, unexpected = model.load_state_dict(state_dict, strict = False)
            if missing:
                log.warning(f'Missing keys when loading classifier: {len(missing)}')
            if unexpected:
                log.warning(f'Unexpected keys when loading classifier: {len(unexpected)}')
        return model

    @torch.no_grad()
    def forward(self, x):
        assert x.shape[-1] == self.img_size, 'Wrong input shape'
        # Expect [0,1] range; rescale if in [-1,1]
        if x.min() < 0:
            assert x.min() >= -1. and x.max() <= 1.
            log.info('Detected input outside [0,1], rescaling from [-1,1] to [0,1]')
            x = (x + 1) / 2
        # Save a debug preview (pre-normalization) if requested
        if self.debug_save_path is not None:
            try:
                torchvision.utils.save_image(x, self.debug_save_path)
                log.info(f"Saved classifier input preview (pre-normalization) to: {self.debug_save_path}")
            except Exception as e:
                log.warning(f"Failed to save debug image to {self.debug_save_path}: {e}")
        x = self.transforms(x)
        logits = self.model(x)
        if self.use_softmax_and_query_label:
            probs = F.softmax(logits, dim = 1)
            return probs[:, self.query_label]
        return logits

