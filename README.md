# A Global Sensitivity Analysis of Traffic Microsimulation Input Parameters on Performance Metrics

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains the code used to run the sensitivity analysis in the paper "A Global Sensitivity Analysis of Traffic Microsimulation Input Parameters on Performance Metrics" by Maxwell Schrader and Joshua Bittle.

## Paper Figure Directory

1. Figure 1: Not included in this repository
2. Figure 2: Not included in this repository
3. Figure 3: `./notebooks/SA1/NSweep.ipynb`
4. Figure 4: `./notebooks/SA1/SA1.ipynb`
5. Figure 5: `./notebooks/SA1/SA1.ipynb`
6. Figure 6: `./notebooks/SA1/SA1.ipynb`
7. Figure 7: `./notebooks/SA2/SA2.ipynb`
8. Figure 8: `./notebooks/SA2/SA2.ipynb`



## Setup & Installation

### Python

1. Create a virtual environment: `python3 -m venv venv`
2. Activate the virtual environment: `source venv/bin/activate`
3. Install the requirements: `pip install -r requirements.txt`

### Additional for Running SA

1. Install [SUMO](https://sumo.dlr.de/docs/Installing.html) and add the `bin` directory to your `PATH` environment variable.
    -  You will also need to set the `SUMO_HOME` environment variable to the path of your SUMO installation.
2. Some of the scrips in `./scripts` require the `jq` command line tool. Install it using your package manager of choice.


### Directory Structure

```
.
├── configs
│   ├── common_blocks
│   └── parameter_sets
│       ├── literature
│       └── paper
├── data
│   ├── LiteratureDefaults
│   │   ├── ...
│   ├── SA1
│   │   ├── ...
│   └── SA2
│       └── ...
├── notebooks
│   ├── SA1
│   └── SA2
├── scripts
├── simulation
│   ├── additional
│   │   ├── detectors
│   │   ├── polygons
│   │   └── signals
│   ├── network
│   └── routes
└── src
    ├── plotting
    └── sa_helpers
```

## Running the Sensitivity Analysis

All sensitivity analyses are ran using the `sumo-pipe` command from the [sumo-pipelines](https://github.com/mschrader15/sumo-pipelines) python library.

They use Ray to parallelize the simulations. The number of parallel simulations can be configured using the ray start command:

```shell
ray start --head --port=6379 --num-cpus=<desired cpu num>"
```

All simulations rely on three environment variables:

1. `SUMO_HOME`: The path to your SUMO installation
2. `PYTHONPATH`: The path to `./src` must be added to the `PYTHONPATH` environment variable.
3. `PROJECT_ROOT`: The path to the root of this project. This is used to find the simulation data & set the output


### SA 1

#### Running the Analysis

The sensitivity analyis is configured by three YAML files: 

1. `./configs/sa1.yaml`
    - Defines the workflow and the Metadata
2. `./configs/common_blocks/blocks.yaml`
    - Defines the blocks used in the workflow
3. `./configs/parameter_sets/paper/sa1.yaml`
    - Definies the Sensitivity Analysis parameters & vehicle distributions

The analysis can be run using the following command:


```shell
export PYTHONPATH="$PYTHONPATH:$PWD";
export PROJECT_ROOT="$PWD";
sumo-pipe ./configs/sa1.yaml ./configs/common_blocks/blocks.yaml ./configs/parameter_sets/paper/sa1.yaml
```

The results of the analysis will be stored according to the `Metadata.output_dir` parameter in the `./config/sa1.yaml` file. The reults must be first processed using `scripts/process-results.py` before they can be analyzed.

```shell
python scripts/process_results.py <path to results directory>
```

You can batch execute this for many simulations with

```shell
for dir in <path to results directory>/**/*; do
    python scripts/process_results.py $dir
done
```

The process results script may take a while to run depending on the number of simulations.

### SA 2

#### Running the Analysis

Like SA 1, the sensitivity analyis is configured by three YAML files:

1. `./configs/sa2.yaml`
    - Defines the workflow and the Metadata
2. `./configs/common_blocks/blocks.yaml`
    - Defines the blocks used in the workflow
3. `./configs/parameter_sets/paper/sa2.yaml`
    - Definies the Sensitivity Analysis parameters & vehicle distributions


The analysis can be run using the following command:

```shell
export PYTHONPATH="$PYTHONPATH:$PWD";
export PROJECT_ROOT="$PWD";
sumo-pipe ./configs/sa2.yaml ./configs/common_blocks/blocks.yaml ./configs/parameter_sets/paper/sa2.yaml
```


The results of the analysis will be stored according to the `Metadata.output_dir` parameter in the `./config/sa2.yaml` file. The reults must be first processed using `scripts/process-results.py` before they can be analyzed.


