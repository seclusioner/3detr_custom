#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh
conda activate 3detr

exec "$@"
