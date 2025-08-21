# Common Parameters
# Please change the BASE_PATH to your actual project directory
export BASE_PATH=`pwd`/scratch
export LGF_PATH="${BASE_PATH}/LGF"
export LOG_PATH="${BASE_PATH}/log"
export SHARD_SIZE=3072

if [ ! -d "$BASE_PATH" ]; then
    echo "Error: The base directory '$BASE_PATH' does not exist." >&2
    exit 1
fi