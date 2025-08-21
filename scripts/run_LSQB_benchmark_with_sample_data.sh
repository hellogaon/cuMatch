#!/bin/bash

set -eu
set -o pipefail

cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd ..

. ./scripts/vars.sh

QUERY_IDS=("1-short" "2" "3" "4" "5" "6" "7" "8" "9")
REPEAT_TIMES=1
TIMEOUT=3650

echo "QUERY_ID | SF | total_counter | kernel_elapsed_time | query_elapsed_time"

for QUERY_ID in ${QUERY_IDS[@]}; do
  for ((i=1; i<=REPEAT_TIMES; i++)); do
    # Optional: clear caches for benchmark (requires root)
    # sync
    # sudo sh -c "echo 1 > /proc/sys/vm/drop_caches"
    # sudo sh -c "echo 2 > /proc/sys/vm/drop_caches"
    # sudo sh -c "echo 3 > /proc/sys/vm/drop_caches"

    grid_name="LSQB_sf0.1"
    grid_dir="./dataset/sample"
    query="./queries/q${QUERY_ID}.json"
    host_buffer_size=21474836480
    device_buffer_size=21474836480
    table_join_flag="n"
    in_memory_mode="n"
    gpu_streaming_mode_flag="y"

    output=$(timeout $TIMEOUT ./build/cuMatch ${grid_name} ${grid_dir} ${query} ${host_buffer_size} ${device_buffer_size} ${table_join_flag} ${in_memory_mode} ${gpu_streaming_mode_flag} || echo "timeout")

    if [[ "$output" == *"timeout"* ]]; then
      echo "${QUERY_ID} | timeout | timeout"
      continue
    fi

    query_elapsed_time=$(echo "$output" | grep "Query elapsed time:" | sed -E 's/.*Query elapsed time: ([0-9]+\.[0-9]+) seconds.*/\1/')
    kernel_elapsed_time=$(echo "$output" | grep "Kernel elapsed time:" | sed -E 's/.*Kernel elapsed time: ([0-9]+\.[0-9]+) seconds.*/\1/')
    total_counter=$(echo "$output" | grep "Total counter:" | sed -E 's/.*Total counter: ([0-9]+).*/\1/')
    
    echo "${QUERY_ID} | 0.1 | ${total_counter} | ${kernel_elapsed_time} | ${query_elapsed_time}"
  done
done

