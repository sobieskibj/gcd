import torch
import monai
import torchvision.transforms as tt
import torch.nn.functional as F

import logging
logging.basicConfig(level = logging.INFO)
log = logging.getLogger(__name__)

def get_densenet264():
    model = monai.networks.nets.DenseNet264(
        spatial_dims = 2, 
        in_channels = 3, 
        out_channels = 14)
    return model

id_to_cls = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 
             'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 
             'Lung Opacity', 'No Finding', 'Pleural Effusion', 
             'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices']

class DenseNetCheXpert(torch.nn.Module):
    def __init__(
            self, 
            img_size,
            path_to_weights, 
            query_label,
            use_probs_and_query_label: bool,
            task: str):
        super().__init__()
        self.classifier = get_densenet264()
        self.img_size = img_size
        self.use_probs_and_query_label = use_probs_and_query_label
        self.query_label = query_label
        self.transforms = tt.Compose([tt.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))])
        self.id_to_cls = id_to_cls
        log.info(f'Query label name: {self.id_to_cls[query_label]}')
        self.task = task
        self.load_weights(path_to_weights)
        self.eval()

    def load_weights(self, path_to_weights):
        log.info(f'Loading checkpoint from {path_to_weights}')
        state_dict = torch.load(path_to_weights, map_location = 'cpu')['state_dict']
        self.load_state_dict(state_dict)

    @torch.no_grad()
    def forward(self, x):
        assert x.shape[-1] == self.img_size, 'Wrong input shape'
        assert x.shape[1] == 1, 'Input should have one channel'
        x = x.repeat(1, 3, 1, 1)
        if x.min() < 0:
            log.info('Detected input with values out of [0, 1] range')
            log.info('Rescaling to [0, 1]')
            x = (x + 1) / 2
        x = self.transforms(x)
        x = self.classifier(x)
        if self.use_probs_and_query_label:
            x = F.sigmoid(x)[:, self.query_label]
        return x