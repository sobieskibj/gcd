'''
Script taken from https://github.com/ServiceNow/beyond-trivial-explanations and adapted
'''

import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as tt
from collections import OrderedDict
from PIL import Image

import logging
logging.basicConfig(level = logging.INFO)
log = logging.getLogger(__name__)

id_to_cls = [
        '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
        'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
        'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
        'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
        'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
        'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
        'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
        'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick',
        'Wearing_Necklace', 'Wearing_Necktie', 'Young']

class Identity(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x

class DenseNet121(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.feat_extract = torchvision.models.densenet121(pretrained=False)
        self.feat_extract.classifier = Identity()
        self.output_size = 1024
    
    def forward(self, x):
        return self.feat_extract(x)

class DenseNet(torch.nn.Module):
    def __init__(
            self, 
            img_size,
            path_to_weights, 
            query_label,
            use_probs_and_query_label: bool,
            task: str):
        super().__init__()
        self.feat_extract = DenseNet121()
        self.classifier = torch.nn.Linear(self.feat_extract.output_size, 40)
        self.img_size = img_size
        self.use_probs_and_query_label = use_probs_and_query_label
        self.query_label = query_label
        self.transforms = tt.Compose([
            tt.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))])
        self.task = task
        self.load_weights(path_to_weights)
        self.eval()

    def load_weights(self, path_to_weights):
        log.info(f'Loading checkpoint from {path_to_weights}')
        ckpt = torch.load(path_to_weights, map_location='cpu')
        if 'feat_extract' in ckpt.keys() and 'classifier' in ckpt.keys():
            assert 'celeba' in path_to_weights
            assert self.img_size == 128
            self.feat_extract.load_state_dict(ckpt['feat_extract'])
            self.classifier.load_state_dict(ckpt['classifier'])
            self.id_to_cls = id_to_cls
        elif 'model_state_dict' in ckpt.keys():
            assert 'celeba-hq' in path_to_weights
            assert self.img_size == 256
            self.classifier = torch.nn.Linear(self.feat_extract.output_size, 3)
            state_dict = ckpt['model_state_dict']
            self.load_state_dict(state_dict)
            self.id_to_cls = ['Male', 'Smile', 'Young']
        log.info(f'Query label name: {self.id_to_cls[self.query_label]}')

    @torch.no_grad()
    def forward(self, x):
        assert x.shape[-1] == self.img_size, 'Wrong input shape'
        # NOTE: Input is required to be in [0, 1] range
        if x.min() < 0:
            assert x.min() >= -1. and x.max() <= 1.
            log.info('Detected input with values out of [0, 1] range')
            log.info(f'Minimum value: {x.min().item()}')
            log.info(f'Maximum value: {x.max().item()}')
            log.info('Rescaling to [0, 1]')
            x = (x + 1) / 2
        x = self.transforms(x)
        x = self.feat_extract(x)
        x = self.classifier(x)
        if self.use_probs_and_query_label:
            x = F.sigmoid(x)[:, self.query_label]
        return x