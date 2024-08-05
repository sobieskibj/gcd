import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
from torchvision.io import read_image

class CheXpertDataset(Dataset):

    def __init__(self, path_data: str, path_labels: str, n_samples: int):
        super().__init__()

        self.paths = self.make_paths(path_data)
        self.labels = self.make_labels(path_labels)
        self.length = min(len(self.paths), n_samples)

    def make_paths(self, path):
        paths = list(Path(path).rglob('*.jpg'))
        return sorted(paths, key = lambda x: int(x.parts[-3][7:]))
    
    def make_labels(self, path):
        return pd.read_csv(path, index_col = 0)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        path = self.paths[index]
        img = read_image(str(path)) / 255
        return img, index