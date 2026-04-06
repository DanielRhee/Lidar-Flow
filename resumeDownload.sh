# A very slightly modified version of this https://argoverse.github.io/user-guide/getting_started.html#downloading-the-data

#!/usr/bin/env bash

export DATASET_NAME="lidar"
export TARGET_DIR="$HOME/persistent/dataset/lidar"

s5cmd --no-sign-request sync \
  --size-only \
  "s3://argoverse/datasets/av2/$DATASET_NAME/*" \
  "$TARGET_DIR"
