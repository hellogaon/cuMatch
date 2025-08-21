#!/bin/bash

set -eu
set -o pipefail

cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd ..

. ./scripts/vars.sh
. ./scripts/LSQB_env.sh

if [ ! -d "${LSQB_RELABELED_DATASET_PATH}" ]; then
    mkdir -p "${LSQB_RELABELED_DATASET_PATH}"
fi

./build/csv_relabeler ${LSQB_DATASET_PATH} ${LSQB_RELABELED_DATASET_PATH} ${LSQB_SCHEMA_FILE} y