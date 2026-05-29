"""Generate test mask zip using a Z-threshold for ground (raster files unavailable)."""
import zipfile
from pathlib import Path

import pandas as pd

from extractSceneflow import RawSweepLoader

# Points with sensor-frame z below this are considered ground (~1.5m below sensor)
GROUND_Z = -1.5


def main():
    loader = RawSweepLoader(Path.home() / 'persistent/dataset', 'lidar', 'test')
    evalInds = list(range(len(loader)))[::5]
    maskFile = Path.home() / 'persistent/djrhee/test_masks.zip'
    print(f'building {len(evalInds)} masks → {maskFile}', flush=True)

    with zipfile.ZipFile(maskFile, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for i, idx in enumerate(evalInds):
            pc0, _, __, (logId, ts0) = loader[idx]
            x, y, z = pc0[:, 0], pc0[:, 1], pc0[:, 2]
            mask = ((x.abs() <= 50) & (y.abs() <= 50) & (z >= GROUND_Z)).numpy().astype(bool)
            df = pd.DataFrame({'mask': mask})
            with zf.open(f'{logId}/{ts0}.feather', 'w') as f:
                df.to_feather(f)
            if (i + 1) % 5000 == 0 or i + 1 == len(evalInds):
                print(f'  {i + 1}/{len(evalInds)}', flush=True)

    print('done')


if __name__ == '__main__':
    main()
