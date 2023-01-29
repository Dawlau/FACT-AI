#! /bin/bash

datasets=("rice_subset")
exponent_values=(1.0 2.0 4.0 6.0 8.0)
alpha_values=(0.1 0.3 0.5 0.7 0.9)

for dataset in ${datasets[@]}; do

    # run fairwalk and deepwalk
    for weighted in "fairwalk" "unweighted"; do
        output_file=results/${dataset}_${weighted}.txt

        if ! [ -f $output_file ]; then
            python main.py --method $weighted --dataset $dataset
        else
            echo $output_file already exists
        fi
    done

    # run crosswalk with different parameters
    for alpha in ${alpha_values[@]}; do
      for exponent in ${exponent_values[@]}; do
        output_file=${data}/${dataset}/embeddings_${method}_d32_${i}
        method=random_walk_5_bndry_${alpha}_exp_${exponent}

        if ! [ -f $output_file ]; then
          python main.py --method $method --dataset $dataset
        else
          echo $output_file already exists
        fi
      done
    done
done