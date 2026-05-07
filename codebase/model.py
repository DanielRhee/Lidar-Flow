import torch
import torch.nn as nn
import spconv.pytorch as spconv
from spconv.pytorch.conv import ConvAlgo

from voxelizer import voxelize

# V100 (sm_70) and older: cumm's implicit_gemm tile selection SIGFPEs; use native rule-book path.
_ALGO = ConvAlgo.Native if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8 else None


# Voxel-union early fusion. f0, f1: [V, 4]  c0, c1: [V, 3] int32 (x, y, z order).
# Returns fu [Vu, 10] = [f0-or-zeros, occ0, f1-or-zeros, occ1], cu [Vu, 3], pc0ToUnion [V0].
def buildUnion(f0, c0, f1, c1):
    device = f0.device
    base = 4096
    h0 = (c0[:, 0].long() * base + c0[:, 1].long()) * base + c0[:, 2].long()
    h1 = (c1[:, 0].long() * base + c1[:, 1].long()) * base + c1[:, 2].long()
    hAll = torch.cat([h0, h1])
    cAll = torch.cat([c0, c1], dim=0)
    uniq, invAll = torch.unique(hAll, return_inverse=True)
    Vu = uniq.shape[0]
    V0 = c0.shape[0]

    cu = torch.zeros((Vu, 3), dtype=torch.int32, device=device)
    cu[invAll] = cAll

    fu = torch.zeros((Vu, 10), dtype=torch.float32, device=device)
    inv0 = invAll[:V0]
    inv1 = invAll[V0:]
    fu[inv0, 0:4] = f0
    fu[inv0, 4] = 1.0
    fu[inv1, 5:9] = f1
    fu[inv1, 9] = 1.0

    return fu, cu, inv0


def subMBlock(inC, outC, key):
    return spconv.SparseSequential(
        spconv.SubMConv3d(inC, outC, 3, padding=1, bias=False, indice_key=key, algo=_ALGO),
        nn.BatchNorm1d(outC),
        nn.ReLU(True),
        spconv.SubMConv3d(outC, outC, 3, padding=1, bias=False, indice_key=key, algo=_ALGO),
        nn.BatchNorm1d(outC),
        nn.ReLU(True),
    )


def downBlock(inC, outC, key):
    return spconv.SparseSequential(
        spconv.SparseConv3d(inC, outC, 3, stride=2, padding=1, bias=False, indice_key=key, algo=_ALGO),
        nn.BatchNorm1d(outC),
        nn.ReLU(True),
    )


def upBlock(inC, outC, key):
    return spconv.SparseSequential(
        spconv.SparseInverseConv3d(inC, outC, 3, bias=False, indice_key=key, algo=_ALGO),
        nn.BatchNorm1d(outC),
        nn.ReLU(True),
    )


# Concat features of two sparse tensors that share coords and row order (guaranteed
# by indice_key round-trips in the U-Net).
def catSparse(a, b):
    return a.replace_feature(torch.cat([a.features, b.features], dim=1))


class SparseFlowNet(nn.Module):
    def __init__(self, inC=10):
        super().__init__()
        self.enc0 = subMBlock(inC, 32, "s0")
        self.down1 = downBlock(32, 64, "d1")
        self.enc1 = subMBlock(64, 64, "s1")
        self.down2 = downBlock(64, 128, "d2")
        self.enc2 = subMBlock(128, 128, "s2")
        self.down3 = downBlock(128, 256, "d3")
        self.bot = subMBlock(256, 256, "s3")
        self.up3 = upBlock(256, 128, "d3")
        self.dec2 = subMBlock(256, 128, "s2")
        self.up2 = upBlock(128, 64, "d2")
        self.dec1 = subMBlock(128, 64, "s1")
        self.up1 = upBlock(64, 32, "d1")
        self.dec0 = subMBlock(64, 32, "s0")
        self.head = spconv.SubMConv3d(32, 3, 1, bias=True, indice_key="head", algo=_ALGO)

    def forward(self, x):
        e0 = self.enc0(x)
        e1 = self.enc1(self.down1(e0))
        e2 = self.enc2(self.down2(e1))
        b = self.bot(self.down3(e2))
        d2 = self.dec2(catSparse(self.up3(b), e2))
        d1 = self.dec1(catSparse(self.up2(d2), e1))
        d0 = self.dec0(catSparse(self.up1(d1), e0))
        return self.head(d0)


# Normalize intensity roughly to XYZ scale. Handles both uint8 [0,255] and [0,1] inputs.
def normalizeIntensity(col):
    if col.numel() == 0:
        return col
    return col / torch.where(col.max() > 2.0, col.new_tensor(255.0), col.new_tensor(1.0))


# End-to-end: voxelize both sweeps, build union, run the network, gather per-point flow.
# Returns predPerPoint [N_inrange, 3] and inRangeMask0 [N].
def runForward(model, pc0, pc1, voxelSize, pointRange, device):
    pc0 = pc0[:, :4].to(device=device, dtype=torch.float32, non_blocking=True)
    pc1 = pc1[:, :4].to(device=device, dtype=torch.float32, non_blocking=True)
    pc0[:, 3] = normalizeIntensity(pc0[:, 3])
    pc1[:, 3] = normalizeIntensity(pc1[:, 3])

    f0, c0, shape, inv0Point, mask0 = voxelize(pc0, voxelSize, pointRange)
    f1, c1, _, _, _ = voxelize(pc1, voxelSize, pointRange)

    fu, cu, pc0ToUnion = buildUnion(f0, c0, f1, c1)

    # spconv indices: [batch, z, y, x] int32
    Vu = cu.shape[0]
    batchCol = torch.zeros((Vu, 1), dtype=torch.int32, device=device)
    idxZyx = torch.stack([cu[:, 2], cu[:, 1], cu[:, 0]], dim=1)
    indices = torch.cat([batchCol, idxZyx], dim=1).contiguous()

    zyxShape = [shape[2], shape[1], shape[0]]

    x = spconv.SparseConvTensor(
        features=fu,
        indices=indices,
        spatial_shape=zyxShape,
        batch_size=1,
    )

    out = model(x)
    assert out.features.shape[0] == Vu, "spconv did not preserve union voxel ordering"

    voxelFlow = out.features
    pointToUnion = pc0ToUnion[inv0Point]
    predPerPoint = voxelFlow[pointToUnion]

    return predPerPoint, mask0
