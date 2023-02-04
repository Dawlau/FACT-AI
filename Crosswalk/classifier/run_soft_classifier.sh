#! /bin/bash -l

datasets=("soft_rice_subset")
exponent_values=(1.0 2.0 4.0 6.0 8.0)
alpha_values=(0.5 0.7 0.9)
c_values=(0.2 0.3 0.35)

# Directory of scritpt
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

for dataset in ${datasets[@]}; do
    for weighted in "fairwalk" "unweighted"; do
        output_file=${SCRIPT_DIR}/results/${dataset}_${weighted}.json

        if ! [ -f "$output_file" ]; then
          echo "Running ${output_file}"
          python main.py --method "$weighted" --dataset "$dataset"
        else
          echo "$output_file" already exists
        fi
    done

    # run crosswalk with different parameters
    for c in ${c_values[@]}; do
      for alpha in ${alpha_values[@]}; do
        for exponent in ${exponent_values[@]}; do
          method=c${c}_random_walk_5_bndry_${alpha}_exp_${exponent}
          output_file=${SCRIPT_DIR}/results/${dataset}_${method}.json

          if ! [ -f "$output_file" ]; then
            echo "Running ${output_file}"
            python main.py --method "$method" --dataset "$dataset"
          else
            echo "$output_file" already exists
          fi
        done
      done
    done
done
