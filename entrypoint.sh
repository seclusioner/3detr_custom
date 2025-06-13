#!/bin/bash

# 初始化 Conda 並自動進入 conda 環境
source /opt/conda/etc/profile.d/conda.sh
conda activate 3detr

# 接著執行 container 啟動時給的參數（例如 /bin/bash）
exec "$@"