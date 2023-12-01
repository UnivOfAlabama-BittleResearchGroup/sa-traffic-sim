# A Global Sensitivity Analysis of Traffic Microsimulation Input Parameters on Performance Metrics

## Overview



----------------
## Setup 

### Python

1. Create a virtual environment: `python3 -m venv venv`
2. Activate the virtual environment: `source venv/bin/activate`
3. Install the requirements: `pip install -r requirements.txt`


### Additional for Running SA

1. Install [SUMO](https://sumo.dlr.de/docs/Installing.html) and add the `bin` directory to your `PATH` environment variable.
    -  You will also need to set the `SUMO_HOME` environment variable to the path of your SUMO installation.
2. Some of the scrips in `./scripts` require the `jq` command line tool. Install it using your package manager of choice.


-------------

## Sensitivity Analysis

All sensitivity analyses are ran using the `sumo-pipe` command from the [sumo-pipelines](https://github.com/mschrader15/sumo-pipelines) python library.

They use Ray to parallelize the simulations. The number of parallel simulations can be configured using the ray start command:

```shell
ray start --head --port=6379 --num-cpus=<desired cpu num>"
```


### SA 1

#### Running the Analysis

The sensitivity analyis is configured by three YAML files: 

1. `./configs/sa1.yaml`
    - Defines the workflow and the Metadata
2. `./configs/common-blocks/blocks.yaml`
    - Defines the blocks used in the workflow
3. `./configs/parameter_sets/paper/sa1.yaml`
    - Definies the Sensitivity Analysis parameters & vehicle distributions

The analysis can be run using the following command:


```shell
sumo-pipe ./configs/sa1.yaml ./configs/common-blocks/blocks.yaml ./configs/parameter_sets/paper/sa1.yaml
```

The results of the analysis will be stored according to the `Metadata.output_dir` parameter in the `./config/sa1.yaml` file. The reults must be first processed using `scripts/process-results.py` before they can be analyzed.

```shell
export PYTHONPATH="$PYTHONPATH:$PWD"; python scripts/process_results.py <path to results directory>
```

You can batch execute this for many simulations with

```shell
for dir in <path to results directory>/**/*; do
    python scripts/process_results.py $dir
done
```

#### Analyzing the Results

The results anaylsis is in 


### SA 2



