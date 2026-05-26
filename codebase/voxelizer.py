import torch

def voxelize(points, voxelSize, pointRange):
    device = points.device
    prCpu = torch.as_tensor(pointRange, dtype=torch.float32)
    Dx, Dy, Dz = torch.floor((prCpu[3:] - prCpu[:3]) / voxelSize).long().tolist()
    spatialShape = (Dx, Dy, Dz)
    pointRange = prCpu.to(device)
    rangeMin = pointRange[:3]
    rangeMax = pointRange[3:]
 
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
 
    return features, coords, spatialShape, inverse, inRangeMask


def saveBevWithFlowPng(coords, spatialShape, pc0, flow, pointRange, outPath):
    import numpy as np

    bev = torch.zeros((spatialShape[0], spatialShape[1]), dtype=torch.float32)
    bev[coords[:, 0].long().cpu(), coords[:, 1].long().cpu()] = 1.0

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor='white')

    ax0 = axes[0]
    ax0.set_facecolor('white')
    ax0.imshow(bev.T.numpy(), origin='lower', cmap='gray_r',
               extent=[pointRange[0], pointRange[3], pointRange[1], pointRange[4]])
    ax0.set_xlabel('x (m)')
    ax0.set_ylabel('y (m)')
    ax0.set_title('Occupied Voxels (BEV)')
    ax0.set_aspect('equal')

    ax1 = axes[1]
    ax1.set_facecolor('white')
    XY_RANGE = 50.0
    ax1.set_xlim(-XY_RANGE, XY_RANGE)
    ax1.set_ylim(-XY_RANGE, XY_RANGE)
    ax1.set_aspect('equal')

    p0 = pc0[:, :3].cpu().numpy()
    ax1.scatter(p0[:, 0], p0[:, 1], s=0.4, c='#888888', alpha=0.5, linewidths=0)

    if flow is not None:
        fv = flow.flow.cpu().numpy()
        dyn = flow.is_dynamic.cpu().numpy().astype(bool)
        inRange = (np.abs(p0[:, 0]) < XY_RANGE) & (np.abs(p0[:, 1]) < XY_RANGE)
        arrowIdx = np.where(dyn & inRange)[0][::5]
        ax1.quiver(
            p0[arrowIdx, 0], p0[arrowIdx, 1],
            fv[arrowIdx, 0], fv[arrowIdx, 1],
            color='#e63946', alpha=0.85,
            scale=1, scale_units='xy',
            width=0.003, headwidth=4, zorder=4,
        )
        ax1.set_title('Scene Flow — Dynamic Points (BEV)')
    else:
        ax1.set_title('Scene Flow (no annotations — test split)')

    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')

    fig.tight_layout()
    fig.savefig(outPath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
 
 
if __name__ == '__main__':
    from pathlib import Path
    import matplotlib.pyplot as plt
    from extractSceneflow import buildLoader, loadAnnotation

    datasetDir = Path.home() / 'persistent' / 'dataset'
    dataset = 'lidar'
    split = 'train'
    idx = 0

    voxelSize = 0.1
    pointRange = [-70.0, -70.0, -3.0, 70.0, 70.0, 3.0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    loader = buildLoader(datasetDir, dataset, split)
    pc0, pc1, flow, sweepUuid = loadAnnotation(loader, idx)
    pc0 = pc0.to(device)
    print(f'input points: {pc0.shape}')

    features, coords, spatialShape, _, _ = voxelize(pc0, voxelSize, pointRange)
    print(f'spatial shape: {list(spatialShape)}, occupied voxels: {features.shape[0]}')

    outPath = Path(__file__).parent / 'voxel_bev.png'
    saveBevWithFlowPng(coords, spatialShape, pc0, flow, pointRange, outPath)
    print(f'saved to {outPath}')
