"""Merge per-fold out-of-fold dumps into one, keeping log identity globally unique.

Each evaluate.py run numbers logIdx from 0 in first-seen order, so the five fold dumps
all start at 0 and would collide. Every log-level statistic downstream -- the paired
bootstrap, conformal's cal/test split, the log-shift measurement -- groups by logIdx,
so a collision would silently merge unrelated logs. Offset each fold by the number of
logs already seen.
"""

import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    tmp = args.out.parent / (args.out.name + ".tmp")
    writer = None
    offset, total = 0, 0
    try:
        for path in args.inputs:
            with ipc.open_file(str(path)) as reader:
                schema = reader.schema
                if writer is None:
                    writer = ipc.new_file(str(tmp), schema)
                nLogs = 0
                for i in range(reader.num_record_batches):
                    b = reader.get_batch(i)
                    logIdx = b.column("logIdx").to_numpy().astype(np.int64) + offset
                    assert logIdx.max() < 65536, "logIdx would overflow uint16"
                    nLogs = max(nLogs, int(logIdx.max()) - offset + 1)
                    cols = {f.name: (logIdx.astype(np.uint16) if f.name == "logIdx"
                                     else b.column(f.name).to_numpy(zero_copy_only=False))
                            for f in schema}
                    batch = pa.record_batch(cols, schema=schema)
                    writer.write_batch(batch)
                    total += batch.num_rows
            print(f"  {path.name}: {nLogs} logs -> global ids {offset}..{offset + nLogs - 1}")
            offset += nLogs
    finally:
        if writer is not None:
            writer.close()
    tmp.rename(args.out)
    print(f"wrote {total:,} points over {offset} logs → {args.out}")


if __name__ == "__main__":
    main()
