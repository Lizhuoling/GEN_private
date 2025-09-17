import pdb
import torch
from torch.utils.data import Dataset

class CustomConcatDataset(Dataset):
    def __init__(self, datasets):
        super(CustomConcatDataset, self).__init__()
        self.datasets = datasets
        self.cumulative_sizes = self._get_cumulative_sizes()

    def _get_cumulative_sizes(self):
        cumulative_sizes = [0]
        for dataset in self.datasets:
            cumulative_sizes.append(cumulative_sizes[-1] + len(dataset))
        return cumulative_sizes

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.__len__():
            raise IndexError("Index out of range")
        dataset_idx = 0
        for i, cumulative_size in enumerate(self.cumulative_sizes[1:], start=1):
            if idx < cumulative_size:
                dataset_idx = i - 1
                break
        local_idx = idx - self.cumulative_sizes[dataset_idx]
        return self.datasets[dataset_idx][local_idx]