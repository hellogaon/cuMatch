#!/bin/bash

set -eu
set -o pipefail

cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd ..

. ./scripts/vars.sh

#  Supports up to MaxNumGPUs GPUs (see include/cu_match/device_defines.cuh)
# Single GPU
docker run -it --name cumatch \
                --runtime=nvidia \
                --gpus '"device=0"' \
                -v ${BASE_PATH}:/root/cuMatch/scratch \
                cumatch

# Multiple GPU
# docker run -it --name cumatch \
#                 --runtime=nvidia \
#                 --gpus 4 \
#                 -v ${BASE_PATH}:/root/cuMatch/scratch \
#                 cumatch