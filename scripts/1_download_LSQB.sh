#!/bin/bash
# [reference] https://repository.surfsara.nl/datasets/cwi/lsqb
# [reference] https://github.com/ldbc/lsqb/blob/main/scripts/download-projected-fk-data-sets.sh

set -eu
set -o pipefail

cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd ..

. ./scripts/vars.sh

if [ ! -d "${LSQB_DOWNLOAD_PATH}" ]; then
    mkdir -p "${LSQB_DOWNLOAD_PATH}"
fi

wget --no-check-certificate -P ${LSQB_DOWNLOAD_PATH} https://repository.surfsara.nl/datasets/cwi/lsqb/files/lsqb-projected/social-network-sf${SF}-projected-fk.tar.zst
tar -I zstd -xf "${LSQB_DOWNLOAD_PATH}/social-network-sf${SF}-projected-fk.tar.zst" -C "${LSQB_DOWNLOAD_PATH}"
