F="${PROJECT_ROOT}/configs/sensitivity_analysis/cars.yaml"

NS=("2", "4", "8", "16", "32", "64", "128", "256", "512", "1024", "2048");
for n in "${NS[@]}"; 
do
    # doing this because I am bad at bash
    export N=$n
    S=$(yq -i e '.Blocks.SobolSequenceConfig.N=env(N)' $F)  # this will edit in place
    sumo-pipe "$F" "${PROJECT_ROOT}/configs/sensitivity_analysis/parameter_sets/paper/cars_paper.yaml" "${PROJECT_ROOT}/configs/sensitivity_analysis/common_blocks/blocks.yaml" 
done