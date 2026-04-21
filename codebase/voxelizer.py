import torch

def voxelize(points, voxelSize, pointRange):
    device = points.device
    pointRange = torch.as_tensor(pointRange, dtype=torch.float32, device=device)
    rangeMin = pointRange[:3]
    rangeMax = pointRange[3:]
 
    spatialShape = torch.floor((rangeMax - rangeMin) / voxelSize).to(torch.long)
    Dx, Dy, Dz = spatialShape.tolist()
 
    xyz = points[:, :3]
    intensity = points[:, 3]
 
    inRangeMask = ((xyz >= rangeMin) & (xyz < rangeMax)).all(dim=1)
    xyz = xyz[inRangeMask]
    intensity = intensity[inRangeMask]
 
    voxelIdx = torch.floor((xyz - rangeMin) / voxelSize).to(torch.long)
    xIdx, yIdx, zIdx = voxelIdx[:, 0], voxelIdx[:, 1], voxelIdx[:, 2]
 
    # Flatten 3D voxel index to 1D for scatter-based aggregation
    flatIdx = xIdx * (Dy * Dz) + yIdx * Dz + zIdx
    uniqueFlat, inverse = torch.unique(flatIdx, return_inverse=True)
    
    numVoxels = uniqueFlat.shape[0] 
    voxelCenters = (voxelIdx.to(torch.float32) + 0.5) * voxelSize + rangeMin
    relXyz = xyz - voxelCenters
    pointFeats = torch.cat([relXyz, intensity.unsqueeze(1)], dim=1)
    featSum = torch.zeros((numVoxels, 4), dtype=torch.float32, device=device)
    featSum.index_add_(0, inverse, pointFeats)
    counts = torch.zeros(numVoxels, dtype=torch.float32, device=device)
    counts.index_add_(0, inverse, torch.ones_like(inverse, dtype=torch.float32))

    features = featSum / counts.unsqueeze(1)
 
    # Recover per-voxel 3D coords from unique flat indices
    zCoord = uniqueFlat % Dz
    yCoord = (uniqueFlat // Dz) % Dy
    xCoord = uniqueFlat // (Dy * Dz)
    coords = torch.stack([xCoord, yCoord, zCoord], dim=1).to(torch.int32)
 
    return features, coords, spatialShape


def saveBevPng(coords, spatialShape, voxelSize, pointRange, outPath):
    bev = torch.zeros((spatialShape[0].item(), spatialShape[1].item()), dtype=torch.float32)
    bev[coords[:, 0].long().cpu(), coords[:, 1].long().cpu()] = 1.0
 
    plt.figure(figsize=(8, 8))
    plt.imshow(bev.T.numpy(), origin='lower', cmap='gray_r',
               extent=[pointRange[0], pointRange[3], pointRange[1], pointRange[4]])
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('Occupied voxels (BEV)')
    plt.tight_layout()
    plt.savefig(outPath, dpi=100)
    plt.close()
 
 
if __name__ == '__main__':
    # Vibecoded idk what the main/visualize stuff does
    from pathlib import Path
    import numpy as np
    import matplotlib.pyplot as plt
    from av2.torch.data_loaders.scene_flow import SceneFlowDataloader
    import pyarrow.feather as feather
 
    datasetDir = Path.home() / 'persistent' / 'data' / 'lidar'
    split = 'train'
    logIdx = 0
    sweepIdx = 0
 
    voxelSize = 0.1
    pointRange = [-70.0, -70.0, -3.0, 70.0, 70.0, 3.0]
 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')
 
    splitDir = datasetDir / split
    logDirs = sorted([d for d in splitDir.iterdir() if d.is_dir()])
    logDir = logDirs[logIdx]
    sweepFiles = sorted((logDir / 'sensors' / 'lidar').glob('*.feather'))
    sweepPath = sweepFiles[sweepIdx]
    print(f'log: {logDir.name}, sweep: {sweepPath.name}')
 
    df = feather.read_feather(sweepPath)
    xyz = torch.from_numpy(np.stack([df['x'].to_numpy(), df['y'].to_numpy(), df['z'].to_numpy()], axis=1)).to(torch.float32)
    intensity = torch.from_numpy(df['intensity'].to_numpy()).to(torch.float32)
    points = torch.cat([xyz, intensity.unsqueeze(1)], dim=1).to(device)
 
    print(f'input points: {points.shape}, dtype={points.dtype}')
 
    features, coords, spatialShape = voxelize(points, voxelSize, pointRange)
 
    print(f'spatial shape (Dx, Dy, Dz): {spatialShape.tolist()}')
    print(f'occupied voxels: {features.shape[0]}')
    totalVoxels = spatialShape.prod().item()
    print(f'sparsity: {100.0 * (1.0 - features.shape[0] / totalVoxels):.4f}%')
    print(f'feature stats: min={features.min().item():.4f}, max={features.max().item():.4f}, mean={features.mean().item():.4f}')
    print(f'coord bounds: x=[{coords[:,0].min().item()}, {coords[:,0].max().item()}], '
          f'y=[{coords[:,1].min().item()}, {coords[:,1].max().item()}], '
          f'z=[{coords[:,2].min().item()}, {coords[:,2].max().item()}]')
 
    outPath = Path(__file__).parent / 'voxel_bev.png'
    saveBevPng(coords, spatialShape, voxelSize, pointRange, outPath)
    print(f'BEV saved to {outPath}')
