# Running parameters
data=../data # Directory of the dataset.
num_workers=10 # Number of parallel processes. (default: 1)
num_walks=80 # Number of random walks to start at each node (default: 10)

datasets=("rice_subset" "twitter")
exponent_values=(1.0 2.0 4.0 6.0 8.0)
alpha_values=(0.1 0.3 0.5 0.7 0.9)

for i in {1..5}; do
  for dataset in ${datasets[@]}; do
    attr_file=$(ls ${data}/${dataset} | grep "\.attr")
    links_file=$(ls ${data}/${dataset} | grep "\.links")

    attr_file=${data}/${dataset}/${attr_file}
    links_file=${data}/${dataset}/${links_file}

    # run fairwalk and deepwalk
    for weighted in "fairwalk" "unweighted"; do
      output_file=${data}/${dataset}/${dataset}.embeddings_${weighted}_d32_${i}
      test_link_file="${data}/${dataset}/${dataset}_${weighted}_${i}_testlinks"
      train_link_file="${data}/${dataset}/${dataset}_${weighted}_${i}_trainlinks"

      if ! [ -f "$output_file" ]; then
        python deepwalk --format edgelist \
                        --input "$links_file" \
                        --max-memory-data-size 0 \
                        --number-walks $num_walks \
                        --representation-size 32 \
                        --walk-length 40 \
                        --window-size 10 \
                        --workers $num_workers \
                        --weighted $weighted  \
                        --output "$output_file" \
                        --sensitive-attr-file "$attr_file" \
                        --test-links 0.1 \
                        --test-links-file "$test_link_file" \
                        --train-links-file "$train_link_file"
      else
        echo "$output_file" already exists
      fi
    done

    # run crosswalk with different parameters
    for alpha in ${alpha_values[@]}; do
      for exponent in ${exponent_values[@]}; do
        method=random_walk_5_bndry_${alpha}_exp_${exponent}
        output_file=${data}/${dataset}/${dataset}.embeddings_${method}_d32_${i}
        test_link_file="${data}/${dataset}/${dataset}_${method}_${i}_testlinks"
        train_link_file="${data}/${dataset}/${dataset}_${method}_${i}_trainlinks"

        if ! [ -f "$output_file" ]; then
          python deepwalk --format edgelist \
                  --input "$links_file" \
                  --max-memory-data-size 0 \
                  --number-walks $num_walks \
                  --representation-size 32 \
                  --walk-length 40 \
                  --window-size 10 \
                  --workers $num_workers \
                  --weighted "$method"  \
                  --output "$output_file" \
                  --sensitive-attr-file "$attr_file" \
                  --test-links 0.1 \
                  --test-links-file "$test_link_file" \
                  --train-links-file "$train_link_file"
        else
          echo "$output_file" already exists
        fi
      done
    done
  done
done
