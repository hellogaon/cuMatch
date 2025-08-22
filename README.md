# cuMatch: A GPU-based Memory-Efficient Worst-case Optimal Join Processing Method for Subgraph Queries with Complex Patterns [SIGMOD'25]

This repository provides the implementation of **cuMatch**, a system that processes subgraph queries on GPUs using the worst-case optimal join (WCOJ) approach. 
It includes scripts and tools to run the full Large-scale Complex Subgraph Query Benchmark (LSQB), which serves as the primary benchmark in our experimental evaluation.
Beyond LSQB, cuMatch also provides a flexible framework that converts arbitrary graph data from CSV into the Labeled Grid Format (LGF) and executes user-defined subgraph queries, including those with complex patterns, in a memory-efficient GPU environment.

Prerequisite
--------

Make sure you have installed all of the following prerequisites on your development machine.

### Run locally
- g++ 7.5.0
- CMake 3.15
- Boost 1.82.0
- CUDA Toolkit 11.6 or later
- Nvidia Driver 510.39.01 or higher

### Run with Docker
- Docker 28.1.1 or later
- CUDA Toolkit 11.6 or later
- Nvidia Driver 510.39.01 or higher
- NVIDIA Container Toolkit

Detailed installation guides for each component can be found [here](https://github.com/hellogaon/cuMatch/tree/main/prerequisite).


Hardware Setting
--------
The hardware specifications used in the paper are shown below.
- **OS:** Ubuntu 20.04
- **CPU:** 16-core 3.0 GHz CPU * 2
- **GPU:** A100 with a capacity of 80 GB
- **Memory:** 1 TB
- **SSD:** PCI-E SSD 6.4 TB * 4


Getting Started Guide
--------
### Run locally

**1. Clone the source code**
```
$ git clone https://github.com/hellogaon/cuMatch.git
$ cd cuMatch
```

**2. Change `CMakeLists.txt` file**
1. Compute Capability setting ([reference](https://developer.nvidia.com/cuda-gpus))
```
# If the installed GPU has a Compute Capability of 8.0,

list(APPEND CUDA_NVCC_FLAGS 
	-gencode arch=compute_80,code=sm_80
	-O3 -std=c++14 -Xcompiler -fopenmp)
```
2) Boost library path setting
```
# If Boost library is installed in ~/local,

set(BOOST_ROOT "~/local")
set(BOOST_INCLUDEDIR "~/local/include")
set(BOOST_LIBRARYDIR "~/local/lib")
```

**3. Build**
```
$ mkdir build
$ cd build
$ cmake ../
$ make -j `nproc`
$ cd ..
```

### Run with Docker
**1. Clone the source code**
```
$ git clone https://github.com/hellogaon/cuMatch.git
$ cd cuMatch
```

**2. Change `CMakeLists.txt` file**
1. Compute Capability setting ([reference](https://developer.nvidia.com/cuda-gpus))
```
# If the installed GPU has a Compute Capability of 8.0,

list(APPEND CUDA_NVCC_FLAGS 
	-gencode arch=compute_80,code=sm_80
	-O3 -std=c++14 -Xcompiler -fopenmp)
```

**3. Build Dockerfile**
```
$ docker build --tag cumatch .
```
**4. Run Docker**
```
# ./scripts/launch_cuMatch_docker.sh
```

Run Sample Queries
--------
The following commands reproduce the data graph and queries shown in Figure 1 of the paper.
The data graph is first converted from CSV to LGF, after which several types of queries (regular, negative, and optional patterns) are executed using `cuMatch`.

```
# Generate the data graph
# ./build/labeled_gridgen <GRAPH_NAME> <CSV_PATH> <OUTPUT_PATH> <SCHEMA_PATH>
$ ./build/labeled_gridgen LGF_sample ./sample/CSV ./sample/LGF ./schema/sample_schema.json

# Run sample queries
$ ./build/cuMatch LGF_sample ./sample/LGF ./queries/sample-q1.json 1 1 n n y # regular pattern
$ ./build/cuMatch LGF_sample ./sample/LGF ./queries/sample-q2.json 1 1 n n y # negative pattern
$ ./build/cuMatch LGF_sample ./sample/LGF ./queries/sample-q3.json 1 1 n n y # optional pattern
```


Evaluate LSQB Benchmark
--------

All datasets and outputs, including logs for each step, are stored under the `BASE_PATH` defined in `./scripts/vars.sh`. For SF=1000, up to 1.5TB of free disk space may be required. Expected logs and final results for all scale factors can be found [here](https://github.com/hellogaon/cuMatch/tree/main/results).

### Step 1. Download & Preprocess Dataset

Download the pre-generated original dataset from the official [LSQB repository](https://github.com/ldbc/lsqb), then relabel vertices sequentially from 0 within each label as part of preprocessing.
```
# Example: To run with with Scale Factor 0.1 (supported SF: 0.1, 1, 10, 100, 1000)

$ export SF=0.1
$ ./scripts/1_1_download_LSQB_dataset.sh
$ ./scripts/1_2_preprocess_LSQB_dataset.sh
```

### Step 2. Generate LGF

```
$ ./scripts/2_generate_LSQB_LGF.sh
```

### Step 3. Run Benchmark

```
$ ./scripts/3_run_LSQB_benchmark.sh
```

### Run All Steps (Full Pipeline)
To execute all steps for each scale factor in one go:
```
$ ./scripts/reset_and_run_all.sh
```


Parameter Descriptions
--------

### Command
```
$ ./build/cuMatch <GRAPH_NAME> <GRAPH_PATH> <QUERY_PATH> \ 
                  <CPU_MEMORY_BUF_SIZE> <GPU_MEMORY_BUF_SIZE> \
                  <TABLE_JOIN_MODE_FLAG> <IN_MEMORY_MODE_FLAG> <GPU_STREAMING_FLAG>
```

### Input data graph & query graph parameter

- **GRAPH_NAME($1):** Graph name. `e.g., LGF_Sample`
- **GRAPH_PATH($2):** Dataset path. `e.g., ./sample/LGF`
- **QUERY_PATH($3):** Query path. `e.g., ./queries/sample-q1.json`

### Memory parameter

- **CPU_MEMORY_BUF_SIZE:GB($4):** The total CPU memory buffer size used to perform the query. `e.g., 20`
- **GPU_MEMORY_BUF_SIZE:GB($5):** The total GPU memory buffer size used to perform the query. `e.g., 20`

### Optimization mode parameter

- **TABLE_JOIN_MODE_FLAG($6):** Whether to use table join mode (Not mentioned in the paper). `(y/n)`
- **IN_MEMORY_MODE_FLAG($7):** Whether to use in-memory mode. `(y/n)`
- **GPU_STREAMING_FLAG($8):** Whether to use GPU streaming mode. `(y/n)`



Contact
--------
If you have any questions, we encourage you to either create Github issues or get in touch directly with Sungwoo Park at sungwoo.park@kaist.ac.kr.