import torch
import numpy as np
from av2.torch.data_loaders.scene_flow import SceneFlowDataloader
import os
from pathlib import Path

# Load a specific annotaiton in the dataset and correct the following sweep's ego motion. 2 modified point clouds as tensors and the flow 
def loadAnnotation(datasetDir, dataset, split, index):
    loader = SceneFlowDataloader(
        root_dir = datasetDir
        dataset_name = dataset
        split_name = split
        num_acumulated_sweeps = 1
        memory_mapped = False
    )

    sweep0, sweep1, ego, flow = loader[index]

    pc0 = sweep0.lidar.as_tensor()

    # transform ego motion 
    pc1Raw = sweep1.lidar.as_tensor()
    sweep1EgoTransform = ego.inverse()
    pc1XYZ = sweep1EgoTrnasform.trnasform_points(pc1Raw[:, :3])
    pc1 = torch.cat([pc1XYZ, pc1_ego1[:, 3:]], dim=1)

    return pc0, pc1, flow

def visualize():
    pass

if __name__ == "__main__":
    datasetDir = Path.home / "persistent" / "data"
    dataset = "lidar"
    split = "train"
    flow = loadAnnotation(datasetDir, dataset, split)
    visualize(flow)
