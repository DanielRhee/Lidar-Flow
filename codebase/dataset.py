import torch
from torch.utils.data import Dataset

from extractSceneflow import buildLoader, loadAnnotation


class SceneFlowDataset(Dataset):
    def __init__(self, datasetDir, dataset, split):
        self.loader = buildLoader(datasetDir, dataset, split)

    def __len__(self):
        return len(self.loader)

    def __getitem__(self, idx):
        return loadAnnotation(self.loader, idx)


def identityCollate(batch):
    return batch[0]
