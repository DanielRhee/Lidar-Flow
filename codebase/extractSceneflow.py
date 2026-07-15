import numpy as np
import pandas as pd
import torch
from av2.torch.data_loaders.scene_flow import SceneFlowDataloader
from pathlib import Path

from paths import DATASET_TYPE, DEFAULT_DATASET, DEFAULT_DATASET_DIR


def _quatToMat(qw, qx, qy, qz):
    return np.array([
        [1-2*(qy**2+qz**2), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw), 1-2*(qx**2+qz**2), 2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1-2*(qx**2+qy**2)]
    ], dtype=np.float64)


class RawSweepLoader:
    # Loads consecutive lidar sweep pairs directly from feather files;
    # no scene flow annotation files required (works with test split).
    def __init__(self, datasetDir, dataset, split):
        # Mirror av2's <root>/<dataset>/sensor/<split> convention.
        splitDir = Path(datasetDir) / dataset / DATASET_TYPE / split
        self._pairs = []   # (logDir, ts0, ts1)
        self._egoCache = {}  # logName -> {timestamp_ns: (R, t)}
        for logDir in sorted(d for d in splitDir.iterdir() if d.is_dir()):
            lidarDir = logDir / 'sensors' / 'lidar'
            timestamps = sorted(int(f.stem) for f in lidarDir.glob('*.feather'))
            for ts0, ts1 in zip(timestamps[:-1], timestamps[1:]):
                self._pairs.append((logDir, ts0, ts1))
            egoDF = pd.read_feather(logDir / 'city_SE3_egovehicle.feather').set_index('timestamp_ns')
            self._egoCache[logDir.name] = {
                ts: (_quatToMat(row.qw, row.qx, row.qy, row.qz),
                     np.array([row.tx_m, row.ty_m, row.tz_m]))
                for ts, row in egoDF.iterrows()
            }

    def __len__(self):
        return len(self._pairs)

    def __getitem__(self, idx):
        logDir, ts0, ts1 = self._pairs[idx]
        pc0Raw = pd.read_feather(logDir / 'sensors/lidar' / f'{ts0}.feather').to_numpy(dtype=np.float32)
        pc1Raw = pd.read_feather(logDir / 'sensors/lidar' / f'{ts1}.feather').to_numpy(dtype=np.float32)

        egoByTs = self._egoCache[logDir.name]
        R0, t0 = egoByTs[ts0]
        R1, t1 = egoByTs[ts1]
        # ego0_SE3_ego1: transforms points from ego1 frame into ego0 frame
        R = (R0.T @ R1).astype(np.float32)
        t = (R0.T @ (t1 - t0)).astype(np.float32)

        pc0 = torch.from_numpy(pc0Raw)[:, :3]
        pc1 = torch.from_numpy(pc1Raw)[:, :3] @ torch.from_numpy(R).T + torch.from_numpy(t)

        # ego1_SE3_ego0, the inverse; submit.py needs it to convert predictions
        # from the ego0 training frame back to the benchmark's convention.
        ego1SE3ego0 = torch.eye(4, dtype=torch.float32)
        ego1SE3ego0[:3, :3] = torch.from_numpy(R.T)
        ego1SE3ego0[:3, 3] = torch.from_numpy(-(R.T @ t))

        return {
            "pc0": pc0,
            "pc1": pc1,
            "ego1SE3ego0": ego1SE3ego0,
            "uuid": (logDir.name, ts0),
        }

def buildLoader(datasetDir, dataset, split):
    return SceneFlowDataloader(
        root_dir=datasetDir,
        dataset_name=dataset,
        split_name=split,
        num_accumulated_sweeps=1,
        memory_mapped=False,
    )

# Load a specific annotation in the dataset and correct the following sweep's ego motion.
def loadAnnotation(loader, index):
    """Return one cache sample: xyz-only fp16 sweeps, flow, and per-point masks.

    Coordinates are stored fp16 because AV2 stores x/y/z as fp16 on disk, so for
    pc0 this is lossless against the source. pc1's xyz is a genuine fp32 result of
    the SE3 compensation; fp16 quantizes it to ~0.06 m at the 70 m range limit,
    which is well under the 0.2 m voxel size.

    is_ground comes from av2's map raster (SceneFlowDataloader builds an
    ArgoverseStaticMap per item and passes it to Sweep.from_rust). It is kept as a
    mask rather than applied here so ground removal stays ablatable.

    Frame convention: av2's GT flow maps pc0 into the *ego1* frame, so a static
    point's GT flow equals the ego motion. Since pc1 is compensated into the ego0
    frame, the flow is brought back to ego0 as well:
        flowEgo0 = ego0_SE3_ego1 . (pc0 + gtFlow) - pc0
    which is 0 for a static point, leaving the network to model only real object
    motion. submit.py inverts this back to the benchmark's convention.
    """
    sweep0, sweep1, ego, flow = loader[index]

    pc0 = sweep0.lidar.as_tensor()[:, :3]

    pc1Raw = sweep1.lidar.as_tensor()
    ego_0_SE3_ego_1 = ego.inverse()
    R = ego_0_SE3_ego_1.rotation.matrix().squeeze(0)
    t = ego_0_SE3_ego_1.translation.squeeze(0)
    pc1 = (pc1Raw[:, :3] @ R.T) + t
    flowEgo0 = ((pc0 + flow.flow) @ R.T + t) - pc0

    return {
        "pc0": pc0.to(torch.float16),
        "pc1": pc1.to(torch.float16),
        "flow": flowEgo0.to(torch.float16),
        # ego1_SE3_ego0 as [4,4]; inverts flowEgo0 back to the benchmark frame.
        "ego1SE3ego0": ego.matrix().squeeze(0).to(torch.float32),
        "isValid": flow.is_valid.to(torch.bool),
        "isDynamic": flow.is_dynamic.to(torch.bool),
        "categoryIndices": flow.category_indices.to(torch.uint8),
        "isGround0": sweep0.is_ground.to(torch.bool),
        "isGround1": sweep1.is_ground.to(torch.bool),
        "uuid": sweep0.sweep_uuid,
    }

# Vibecoded visualization because it's not that important
def visualize(sample):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    def to_np(t):
        return t.detach().cpu().float().numpy()

    p0 = to_np(sample["pc0"])   # [N, 3]
    p1 = to_np(sample["pc1"])   # [M, 3]

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    fig.patch.set_facecolor("#0d0d0d")

    XY_RANGE = 50.0

    for ax in axes:
        ax.set_facecolor("#0d0d0d")
        ax.set_aspect("equal")
        ax.set_xlim(-XY_RANGE, XY_RANGE)
        ax.set_ylim(-XY_RANGE, XY_RANGE)
        ax.tick_params(colors="white")
        ax.set_xlabel("y — lateral (m)", color="white")
        ax.set_ylabel("x — forward (m)", color="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

    # ── Panel 1: sweep0 ───────────────────────────────────────────────────────
    axes[0].scatter(p0[:, 1], p0[:, 0], s=0.4, c="#aaaaaa", alpha=0.6, linewidths=0)
    axes[0].plot(0, 0, "w+", markersize=10, markeredgewidth=2, zorder=5)
    axes[0].set_title("sweep0  (t=0)", color="white", pad=8)

    # ── Panel 2: sweep1 ego-compensated ──────────────────────────────────────
    axes[1].scatter(p1[:, 1], p1[:, 0], s=0.4, c="#4488ff", alpha=0.6, linewidths=0)
    axes[1].plot(0, 0, "w+", markersize=10, markeredgewidth=2, zorder=5)
    axes[1].set_title("sweep1  (ego-compensated into ego0 frame)", color="white", pad=8)

    # ── Panel 3: overlay + flow arrows ───────────────────────────────────────
    ax = axes[2]
    ax.scatter(p0[:, 1], p0[:, 0], s=0.4, c="#aaaaaa", alpha=0.4, linewidths=0, label="sweep0")
    ax.scatter(p1[:, 1], p1[:, 0], s=0.4, c="#4488ff", alpha=0.2, linewidths=0, label="sweep1")

    if sample.get("flow") is not None:
        fv  = to_np(sample["flow"])                      # [N, 3]
        dyn = to_np(sample["isDynamic"]).astype(bool)   # [N]

        in_range   = (np.abs(p0[:, 0]) < XY_RANGE) & (np.abs(p0[:, 1]) < XY_RANGE)
        arrow_mask = dyn & in_range
        arrow_idx  = np.where(arrow_mask)[0][::5]  # every 5th to avoid clutter

        ax.quiver(
            p0[arrow_idx, 1], p0[arrow_idx, 0],    # origin: y, x (BEV convention)
            fv[arrow_idx, 1], fv[arrow_idx, 0],    # direction: fy, fx
            color="yellow", alpha=0.8,
            scale=1, scale_units="xy",
            width=0.003, headwidth=4, zorder=4,
        )
        ax.set_title("overlay + flow arrows (dynamic only)", color="white", pad=8)
    else:
        ax.set_title("overlay (no flow — test split)", color="white", pad=8)

    ax.plot(0, 0, "w+", markersize=10, markeredgewidth=2, zorder=5)
    ax.legend(loc="upper right", fontsize=8,
              facecolor="#222", edgecolor="#555", labelcolor="white")

    out_path = Path.cwd() / "sceneflow_viz.png"
    plt.tight_layout(pad=1.5)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[info] Saved → {out_path}")


if __name__ == "__main__":
    datasetDir = DEFAULT_DATASET_DIR
    dataset = DEFAULT_DATASET
    split = "train"
    index = 0

    loader = buildLoader(datasetDir, dataset, split)
    visualize(loadAnnotation(loader, index))
