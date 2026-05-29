import numpy as np
import pandas as pd
import torch
from av2.torch.data_loaders.scene_flow import SceneFlowDataloader
from pathlib import Path


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
        splitDir = Path(datasetDir) / dataset / split
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

        pc0 = torch.from_numpy(pc0Raw)
        pc1_t = torch.from_numpy(pc1Raw)
        pc1XYZ = pc1_t[:, :3] @ torch.from_numpy(R).T + torch.from_numpy(t)
        pc1 = torch.cat([pc1XYZ, pc1_t[:, 3:]], dim=1)

        return pc0, pc1, None, (logDir.name, ts0)

def buildLoader(datasetDir, dataset, split):
    return SceneFlowDataloader(
        root_dir=datasetDir,
        dataset_name=dataset,
        split_name=split,
        num_accumulated_sweeps=1,
        memory_mapped=False,
    )

# Load a specific annotaiton in the dataset and correct the following sweep's ego motion. 2 modified point clouds as tensors and the flow 
def loadAnnotation(loader, index):
    sweep0, sweep1, ego, flow = loader[index]

    pc0 = sweep0.lidar.as_tensor()

    pc1Raw = sweep1.lidar.as_tensor()
    ego_0_SE3_ego_1 = ego.inverse()
    R = ego_0_SE3_ego_1.rotation.matrix().squeeze(0)
    t = ego_0_SE3_ego_1.translation.squeeze(0)
    pc1XYZ = (pc1Raw[:, :3] @ R.T) + t
    pc1 = torch.cat([pc1XYZ, pc1Raw[:, 3:]], dim=1)

    return pc0, pc1, flow, sweep0.sweep_uuid

# Vibecoded visualization because it's not that important
def visualize(pc0, pc1, flow):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    def to_np(t):
        return t.detach().cpu().float().numpy()

    p0 = to_np(pc0)   # [N, 7]
    p1 = to_np(pc1)   # [M, 7]

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

    if flow is not None:
        fv  = to_np(flow.flow)                      # [N, 3]
        dyn = to_np(flow.is_dynamic).astype(bool)  # [N]

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
    datasetDir = Path.home() / "persistent"
    dataset = "data"
    split = "train"
    index = 0

    loader = buildLoader(datasetDir, dataset, split)
    pc0, pc1, flow, _ = loadAnnotation(loader, index)
    visualize(pc0, pc1, flow)
