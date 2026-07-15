"""Generate test mask zip using a Z-threshold for ground (raster files unavailable)."""
import zipfile

import pandas as pd

from extractSceneflow import RawSweepLoader
from paths import DEFAULT_DATASET, DEFAULT_DATASET_DIR, DEFAULT_RUNS_DIR

# Points with sensor-frame z below this are considered ground (~1.5m below sensor)
GROUND_Z = -1.5


def main():
    loader = RawSweepLoader(DEFAULT_DATASET_DIR, DEFAULT_DATASET, 'test')
    evalInds = list(range(len(loader)))[::5]
    maskFile = DEFAULT_RUNS_DIR / 'test_masks.zip'
    maskFile.parent.mkdir(parents=True, exist_ok=True)
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
