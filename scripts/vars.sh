# Common Parameters
# Please change the BASE_PATH to your actual project directory
export BASE_PATH=`pwd`/scratch
export LGF_PATH="${BASE_PATH}/LGF"
export LOG_PATH="${BASE_PATH}/LOG"
export SHARD_SIZE=3072

if [ ! -d "$BASE_PATH" ]; then
    echo "Error: The base directory '$BASE_PATH' does not exist." >&2
    exit 1
fi

# LSQB Dataset Configuration
export LSQB_DOWNLOAD_PATH="${BASE_PATH}/LSQB"
export LSQB_DATASET_PATH="${BASE_PATH}/LSQB/social-network-sf${SF}-projected-fk"
export LSQB_RELABELED_DATASET_PATH="${BASE_PATH}/LSQB/social-network-sf${SF}-projected-fk-relabeled"
export LSQB_LGF_NAME="LSQB_sf${SF}"
export LSQB_SCHEMA_FILE="./schema/LSQB_schema.json"
