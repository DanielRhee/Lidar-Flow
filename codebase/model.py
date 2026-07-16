import math
import torch
import torch.nn as nn
import spconv.pytorch as spconv
from spconv.pytorch.conv import ConvAlgo

from voxelizer import voxelize

# Allegedly fixes cuda versioning issues (according to gpt)
_ALGO = ConvAlgo.Native if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8 else None


# Voxel combining/fusion stuff
def buildUnion(f0, c0, f1, c1):
    device = f0.device
    # Safe while the grid stays under 4096 per axis: 700x700x30 at voxelSize 0.2,
    # 1400x1400x60 at 0.1. Collides silently below ~0.034 m.
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

    # Per sweep: 3 mean rel-xyz channels + 1 occupancy flag.
    fu = torch.zeros((Vu, 8), dtype=torch.float32, device=device)
    inv0 = invAll[:V0]
    inv1 = invAll[V0:]
    fu[inv0, 0:3] = f0
    fu[inv0, 3] = 1.0
    fu[inv1, 4:7] = f1
    fu[inv1, 7] = 1.0

    return fu, cu, inv0


_NORM_GROUPS = 8  # divides every block width here (32, 64, 128, 256)


def norm(c):
    # GroupNorm, not BatchNorm: it has no running-stat buffers, so train and eval
    # normalize identically. BatchNorm's running stats were being poisoned by rare
    # pathological batches (running_var -> NaN in deep layers), which broke eval /
    # inference while train-mode limped on. GroupNorm removes that failure mode
    # entirely. spconv routes it onto the [numVoxels, C] feature tensor.
    return nn.GroupNorm(_NORM_GROUPS, c)


def subMBlock(inC, outC, key):
    return spconv.SparseSequential(
        spconv.SubMConv3d(inC, outC, 3, padding=1, bias=False, indice_key=key, algo=_ALGO),
        norm(outC),
        nn.ReLU(True),
        spconv.SubMConv3d(outC, outC, 3, padding=1, bias=False, indice_key=key, algo=_ALGO),
        norm(outC),
        nn.ReLU(True),
    )


def downBlock(inC, outC, key):
    return spconv.SparseSequential(
        spconv.SparseConv3d(inC, outC, 3, stride=2, padding=1, bias=False, indice_key=key, algo=_ALGO),
        norm(outC),
        nn.ReLU(True),
    )


def upBlock(inC, outC, key):
    return spconv.SparseSequential(
        spconv.SparseInverseConv3d(inC, outC, 3, bias=False, indice_key=key, algo=_ALGO),
        norm(outC),
        nn.ReLU(True),
    )


# Concat sparse tensors
def catSparse(a, b):
    return a.replace_feature(torch.cat([a.features, b.features], dim=1))


class SparseFlowNet(nn.Module):
    def __init__(self, inC=8):
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
        # Per-voxel flow is constant across every point in a voxel, which floors
        # EPE at the voxel quantization scale. Refine it per point from the voxel
        # feature plus the point's offset within its voxel. Zero-init the last
        # layer so delta starts at exactly 0 and this is a true refinement.
        #
        # Sized generously on purpose: training is kernel-launch-bound at
        # batch_size=1 (~33% GPU, 0.39 GB of 16 GB), so extra width here costs
        # almost no wall-clock, and per-point refinement is where the accuracy
        # the voxel grid throws away has to come back.
        self.refineHead = nn.Sequential(
            nn.Linear(32 + 3, 128),
            nn.ReLU(True),
            nn.Linear(128, 128),
            nn.ReLU(True),
            nn.Linear(128, 3),
        )
        nn.init.zeros_(self.refineHead[4].weight)
        nn.init.zeros_(self.refineHead[4].bias)
        self.uncertaintyHead = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(True),
            nn.Linear(32, 1),
        )
        nn.init.zeros_(self.uncertaintyHead[2].weight)
        nn.init.constant_(self.uncertaintyHead[2].bias, math.log(0.1))

    def forwardFeatures(self, x):
        e0 = self.enc0(x)
        e1 = self.enc1(self.down1(e0))
        e2 = self.enc2(self.down2(e1))
        b = self.bot(self.down3(e2))
        d2 = self.dec2(catSparse(self.up3(b), e2))
        d1 = self.dec1(catSparse(self.up2(d2), e1))
        return self.dec0(catSparse(self.up1(d1), e0))

    def forward(self, x):
        d0 = self.forwardFeatures(x)
        flowOut = self.head(d0)
        logVar = torch.clamp(self.uncertaintyHead(d0.features), -10.0, 5.0)
        return flowOut, logVar


# end to end part that gets per point flow and runs everything above
def runForwardFeatures(model, pc0, pc1, voxelSize, pointRange, device):
    pc0 = pc0[:, :3].to(device=device, dtype=torch.float32, non_blocking=True)
    pc1 = pc1[:, :3].to(device=device, dtype=torch.float32, non_blocking=True)

    f0, c0, shape, inv0Point, mask0, rel0Point = voxelize(pc0, voxelSize, pointRange)
    f1, c1, _, _, _, _ = voxelize(pc1, voxelSize, pointRange)

    fu, cu, pc0ToUnion = buildUnion(f0, c0, f1, c1)

    # indexed [batch, z, y, x] they r int32
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

    d0 = model.forwardFeatures(x)
    assert d0.features.shape[0] == Vu, "spconv did not preserve union voxel ordering" # boooo
    return d0, pc0ToUnion, inv0Point, mask0, rel0Point


def runForwardHeads(model, d0, pc0ToUnion, inv0Point, rel0Point):
    flowOut = model.head(d0)
    logVar = torch.clamp(model.uncertaintyHead(d0.features), -10.0, 5.0)

    pointToUnion = pc0ToUnion[inv0Point]
    voxelFlow = flowOut.features[pointToUnion]
    # Break the per-voxel tie: every point in a voxel shares voxelFlow, so add a
    # per-point delta conditioned on where the point sits inside its voxel.
    delta = model.refineHead(torch.cat([d0.features[pointToUnion], rel0Point.to(d0.features.dtype)], dim=1))
    predPerPoint = voxelFlow + delta
    predLogVarPerPoint = logVar[pointToUnion].squeeze(-1)
    return predPerPoint, predLogVarPerPoint


def runForward(model, pc0, pc1, voxelSize, pointRange, device):
    d0, pc0ToUnion, inv0Point, mask0, rel0Point = runForwardFeatures(model, pc0, pc1, voxelSize, pointRange, device)
    predPerPoint, predLogVarPerPoint = runForwardHeads(model, d0, pc0ToUnion, inv0Point, rel0Point)
    return predPerPoint, predLogVarPerPoint, mask0
