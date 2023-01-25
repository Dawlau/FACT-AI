# Running parameters
data=../data # Directory of the dataset.
num_workers=5 # Number of parallel processes. (default: 1)
num_walks=80 # Number of random walks to start at each node (default: 10)

datasets=("rice_subset" "synth2" "synth3" "synthetic_3layers" "twitter")
exponent_values=(1.0 2.0 4.0 5.0 8.0)
alpha_values=(0.1 0.5 0.7 0.9)

for i in {1..5}; do
  for dataset in ${datasets[@]}; do
    attr_file=$(ls ${data}/${dataset} | grep ".attr")
    links_file=$(ls ${data}/${dataset} | grep ".links")

    attr_file=${data}/${dataset}/${attr_file}
    links_file=${data}/${dataset}/${links_file}

    # run fairwalk and deepwalk
    for weighted in "fairwalk" "unweighted"; do
      output_file=${data}/${dataset}/embeddings_${weighted}_d32_${i}

      if ! [ -f $output_file ]; then
        python deepwalk --format edgelist \
                        --input $links_file \
                        --max-memory-data-size 0 \
                        --number-walks $num_walks \
                        --representation-size 32 \
                        --walk-length 40 \
                        --window-size 10 \
                        --workers $num_workers \
                        --weighted $weighted  \
                        --output $output_file \
                        --sensitive-attr-file $attr_file
      else
        echo $output_file already exists
      fi
    done

    # run crosswalk with different parameters
    for alpha in ${alpha_values[@]}; do
      for exponent in ${exponent_values[@]}; do
        output_file=${data}/${dataset}/embeddings_random_walk_${alpha}_bndry_0.1_exp_${exponent}_d32_${i}

        if ! [ -f $output_file ]; then
          python deepwalk --format edgelist \
                  --input $links_file \
                  --max-memory-data-size 0 \
                  --number-walks $num_walks \
                  --representation-size 32 \
                  --walk-length 40 \
                  --window-size 10 \
                  --workers $num_workers \
                  --weighted $weighted  \
                  --output $output_file \
                  --sensitive-attr-file $attr_file
        else
          echo $output_file already exists
        fi
      done
    done
  done
done