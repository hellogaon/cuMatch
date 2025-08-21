#!/bin/bash

set -eu
set -o pipefail

cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd ..

. ./scripts/vars.sh
. ./scripts/LSQB_env.sh

if [ ! -d "${LGF_PATH}" ]; then
    mkdir -p "${LGF_PATH}"
fi

./build/labeled_gridgen ${LSQB_LGF_NAME} ${LSQB_RELABELED_DATASET_PATH} ${LGF_PATH} ${LSQB_SCHEMA_FILE} ${SHARD_SIZE} y n y