import torch
import numpy as np
from av2.torch.data_loaders.scene_flow import SceneFlowDataloader
import os
from pathlib import Path

# Load a specific annotaiton in the dataset and correct the following sweep's ego motion. 2 modified point clouds as tensors and the flow 
def loadAnnotation(datasetDir, dataset, split, index):
    loader = SceneFlowDataloader(
        root_dir = datasetDir,
        dataset_name = dataset,
        split_name = split,
        num_accumulated_sweeps = 1,
        memory_mapped = False
    )

    sweep0, sweep1, ego, flow = loader[index]

    pc0 = sweep0.lidar.as_tensor()

    # transform ego motion 
    pc1Raw = sweep1.lidar.as_tensor()
    sweep1EgoTransform = ego.inverse()
    pc1XYZ = sweep1EgoTransform.transform_points(pc1Raw[:, :3])
    pc1 = torch.cat([pc1XYZ, pc1Raw[:, 3:]], dim=1)

    return pc0, pc1, flow

def visualize(pc0, pc1, flow):
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
 
    # ── Panel 1: sweep0 ────────────────────────────────────────────────────────
    axes[0].scatter(p0[:, 1], p0[:, 0], s=0.4, c="#aaaaaa", alpha=0.6, linewidths=0)
    axes[0].plot(0, 0, "w+", markersize=10, markeredgewidth=2, zorder=5)
    axes[0].set_title("sweep0  (t=0)", color="white", pad=8)
 
    # ── Panel 2: sweep1 ego-compensated ───────────────────────────────────────
    axes[1].scatter(p1[:, 1], p1[:, 0], s=0.4, c="#4488ff", alpha=0.6, linewidths=0)
    axes[1].plot(0, 0, "w+", markersize=10, markeredgewidth=2, zorder=5)
    axes[1].set_title("sweep1  (ego-compensated into ego0 frame)", color="white", pad=8)
 
    # ── Panel 3: overlay + flow arrows ────────────────────────────────────────
    ax = axes[2]
    ax.scatter(p0[:, 1], p0[:, 0], s=0.4, c="#aaaaaa", alpha=0.4, linewidths=0, label="sweep0")
    ax.scatter(p1[:, 1], p1[:, 0], s=0.4, c="#4488ff", alpha=0.2, linewidths=0, label="sweep1")
 
    if flow is not None:
        fv  = to_np(flow.flow)                       # [N, 3]
        dyn = to_np(flow.is_dynamic).astype(bool)   # [N]
 
        # only draw arrows on dynamic points within BEV range
        in_range  = (np.abs(p0[:, 0]) < XY_RANGE) & (np.abs(p0[:, 1]) < XY_RANGE)
        arrow_mask = dyn & in_range
        arrow_idx  = np.where(arrow_mask)[0][::5]   # every 5th to avoid clutter
 
        ax.quiver(
            p0[arrow_idx, 1], p0[arrow_idx, 0],     # origin: y, x (BEV convention)
            fv[arrow_idx, 1], fv[arrow_idx, 0],     # direction: fy, fx
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
    datasetDir = Path.home / "persistent" / "data"
    dataset = "lidar"
    split = "train"
    index = 0
    
    pc0, pc1, flow = loadAnnotation(datasetDir, dataset, split)
    visualize(pc0, pc1, flow)
