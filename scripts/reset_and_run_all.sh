#!/bin/bash

set -eu
set -o pipefail

cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd ..

. ./scripts/vars.sh

SFs=( 0.1 ) # Test
# SFs=( 100 1000 ) # In paper
# SFs=( 1 10 100 1000 ) # All

shopt -s nullglob
files=("${BASE_PATH}"/*)

non_gitkeep_files=()
for f in "${files[@]}"; do
    if [ "$(basename "$f")" != ".gitkeep" ]; then
        non_gitkeep_files+=("$f")
    fi
done

if [ ${#non_gitkeep_files[@]} -ne 0 ]; then
    echo "Directory BASE_PATH=${BASE_PATH} is not empty."
    echo "Please manually clean it before running this script."
    exit 1
fi

if [ ! -d "${LOG_PATH}" ]; then
    mkdir -p "${LOG_PATH}"
fi

for SF in ${SFs[@]}; do
  export SF=$SF
  . ./scripts/LSQB_env.sh

  ./scripts/1_1_download_LSQB_dataset.sh 2>&1 | tee "${LOG_PATH}/1_1_sf${SF}.log"
  ./scripts/1_2_preprocess_LSQB_dataset.sh 2>&1 | tee "${LOG_PATH}/1_2_sf${SF}.log"
  ./scripts/2_generate_LSQB_LGF.sh 2>&1 | tee "${LOG_PATH}/2_sf${SF}.log"
  ./scripts/3_run_LSQB_benchmark.sh 2>&1 | tee "${LOG_PATH}/3_sf${SF}.log"

done