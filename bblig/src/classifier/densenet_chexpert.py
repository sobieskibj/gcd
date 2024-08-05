import torch
import monai
import torchvision.transforms as tt
import torch.nn.functional as F

from .base import ClassifierBase

def get_densenet264():
    return monai.networks.nets.DenseNet264(
        spatial_dims = 2, in_channels = 3, out_channels = 14)

class DenseNetCheXpert(ClassifierBase):
    id_to_cls = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 
                'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 
                'Lung Opacity', 'No Finding', 'Pleural Effusion', 
                'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices']

    def __init__(self, path_ckpt: str):
        super().__init__()
        self.classifier = get_densenet264()
        self.transforms = tt.Compose([tt.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))])
        self.load_ckpt(path_ckpt)
        self.eval()

    def load_ckpt(self, path_ckpt):
        state_dict = torch.load(path_ckpt, map_location = 'cpu')['state_dict']
        self.load_state_dict(state_dict)

    def forward(self, x):
        x = self.transforms(x)
        x = self.classifier(x)
        return x

    def pred_prob(self, x):
        x = self(x)
        return F.sigmoid(x)

    def pred_class(self, x, class_id):
        return (self(x)[:, class_id] > 0.).int()