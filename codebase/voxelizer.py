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
