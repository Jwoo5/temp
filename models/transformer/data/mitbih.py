from torch.utils.data import Dataset, DataLoader

import pandas as pd

class MITBIHDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.dataset = pd.read_csv(path, header = None).to_numpy()
    
    def __len__(self):
        return self.dataset.shape[0]
    
    def __getitem__(self, index):
        return self.dataset[index][:187], self.dataset[index][187]