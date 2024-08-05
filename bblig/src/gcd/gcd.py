import json
import torch
from torch import nn
from pathlib import Path

class GCD(nn.Module):

    def __init__(self, path_dir: str):
        super(GCD, self).__init__()
        self.setup(path_dir)
    
    def setup(self, path_dir):
        path_dir = Path(path_dir)

        # tensor indicating the direction
        dir = torch.load(path_dir / 'dir.pt')
        self.register_buffer('dir', dir)

        with open(path_dir / 'info.json', 'r') as f:
            info_dict = json.loads(f.read())
        
        # label id from the classifier
        self.target_id = info_dict['target_id']
        
        # initial value of the classifier's predictions
        self.init_class = info_dict['init_class']