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


def saveBevWithFlowPng(coords, spatialShape, pc0, pc1, pointRange, outPath):
    import numpy as np
    from scipy.spatial import KDTree

    bev = torch.zeros((spatialShape[0], spatialShape[1]), dtype=torch.float32)
    bev[coords[:, 0].long().cpu(), coords[:, 1].long().cpu()] = 1.0

    fig, axes = plt.subplots(1, 3, figsize=(24, 7), facecolor='white')
    XY_RANGE = 50.0

    # Panel 1: occupied voxels
    ax0 = axes[0]
    ax0.set_facecolor('white')
    ax0.imshow(bev.T.numpy(), origin='lower', cmap='gray_r',
               extent=[pointRange[0], pointRange[3], pointRange[1], pointRange[4]])
    ax0.set_xlabel('x (m)')
    ax0.set_ylabel('y (m)')
    ax0.set_title('Occupied Voxels (BEV)')
    ax0.set_aspect('equal')

    p0 = pc0[:, :3].cpu().numpy()
    p1 = pc1[:, :3].cpu().numpy() if pc1 is not None else None

    # Panel 2: sweep overlay
    ax1 = axes[1]
    ax1.set_facecolor('white')
    ax1.set_xlim(-XY_RANGE, XY_RANGE)
    ax1.set_ylim(-XY_RANGE, XY_RANGE)
    ax1.set_aspect('equal')
    ax1.scatter(p0[:, 0], p0[:, 1], s=0.3, c='#333333', alpha=0.5, linewidths=0, label='sweep $t$')
    if p1 is not None:
        ax1.scatter(p1[:, 0], p1[:, 1], s=0.3, c='#4488ff', alpha=0.4, linewidths=0, label='sweep $t+1$')
        ax1.legend(loc='upper right', markerscale=8, framealpha=0.9)
        ax1.set_title('Consecutive Sweeps Overlay (BEV)')
    else:
        ax1.set_title('Point Cloud (BEV)')
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')

    # Panel 3: flow arrows
    ax2 = axes[2]
    ax2.set_facecolor('white')
    ax2.set_xlim(-XY_RANGE, XY_RANGE)
    ax2.set_ylim(-XY_RANGE, XY_RANGE)
    ax2.set_aspect('equal')
    ax2.scatter(p0[:, 0], p0[:, 1], s=0.3, c='#333333', alpha=0.4, linewidths=0)
    if p1 is not None:
        # Approximate flow: nearest-neighbor match, ego-motion removed via median subtraction
        inRange = (np.abs(p0[:, 0]) < XY_RANGE) & (np.abs(p0[:, 1]) < XY_RANGE)
        p0r = p0[inRange]
        p0s = p0r[::max(1, len(p0r) // 3000)]

        tree = KDTree(p1[:, :2])
        _, nnIdx = tree.query(p0s[:, :2])
        flowVec = p1[nnIdx, :2] - p0s[:, :2]

        relFlow = flowVec - np.median(flowVec, axis=0)
        mag = np.linalg.norm(relFlow, axis=1)
        movingMask = (mag > 0.5) & (mag < 15.0)

        if movingMask.any():
            ax2.quiver(
                p0s[movingMask, 0], p0s[movingMask, 1],
                relFlow[movingMask, 0], relFlow[movingMask, 1],
                color='#ffdd00', alpha=0.9, scale=1, scale_units='xy',
                width=0.004, headwidth=4, zorder=5,
            )
        ax2.set_title('Approximate Scene Flow (BEV)')
    else:
        ax2.set_title('Scene Flow (BEV)')
    ax2.set_xlabel('x (m)')
    ax2.set_ylabel('y (m)')

    fig.tight_layout()
    fig.savefig(outPath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
 
 
if __name__ == '__main__':
    from pathlib import Path
    import numpy as np
    import matplotlib.pyplot as plt
    import pyarrow.feather as feather

    datasetDir = Path.home() / 'persistent' / 'dataset' / 'lidar'
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
    pc0 = torch.cat([xyz, intensity.unsqueeze(1)], dim=1).to(device)
    print(f'input points: {pc0.shape}')

    features, coords, spatialShape, _, _ = voxelize(pc0, voxelSize, pointRange)
    print(f'spatial shape: {list(spatialShape)}, occupied voxels: {features.shape[0]}')

    sweepPath1 = sweepFiles[sweepIdx + 1]
    df1 = feather.read_feather(sweepPath1)
    xyz1 = torch.from_numpy(np.stack([df1['x'].to_numpy(), df1['y'].to_numpy(), df1['z'].to_numpy()], axis=1)).to(torch.float32)
    intensity1 = torch.from_numpy(df1['intensity'].to_numpy()).to(torch.float32)
    pc1 = torch.cat([xyz1, intensity1.unsqueeze(1)], dim=1)

    outPath = Path(__file__).parent / 'voxel_bev.png'
    saveBevWithFlowPng(coords, spatialShape, pc0, pc1, pointRange, outPath)
    print(f'saved to {outPath}')
