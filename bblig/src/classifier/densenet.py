import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as tt

from .base import ClassifierBase

class Identity(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class DenseNet121(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.feat_extract = torchvision.models.densenet121(weights = None)
        self.feat_extract.classifier = Identity()
        self.output_size = 1024
    
    def forward(self, x):
        return self.feat_extract(x)

class DenseNet(ClassifierBase):
    id_to_cls = ['male', 'smile', 'young']

    def __init__(self, path_ckpt: str):
        super().__init__()
        
        self.feat_extract = DenseNet121()
        self.classifier = torch.nn.Linear(self.feat_extract.output_size, 3)
        self.transforms = tt.Compose([tt.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))])
        self.load_ckpt(path_ckpt)
        self.eval()

    def load_ckpt(self, path_ckpt):
        ckpt = torch.load(path_ckpt, map_location = 'cpu')
        self.load_state_dict(ckpt['model_state_dict'], strict = False)

    def forward(self, x):
        # NOTE: Input is required to be in [0, 1] range
        x = self.transforms(x)
        x = self.feat_extract(x)
        x = self.classifier(x)
        return x
    
    def pred_prob(self, x):
        x = self(x)
        return F.sigmoid(x)

    def pred_class(self, x, class_id):
        return (self(x)[:, class_id] > 0.).int()