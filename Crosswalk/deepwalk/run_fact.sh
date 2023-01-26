# The dataset to be used in deepwalk to generate node representations.
#dataset='twitter'
dataset='rice_subset'

# Running parameters
data=../data # Directory of the dataset.
d=32 # Dimensionality of the latent space.
num_workers=24 # Number of parallel processes. (default: 1)
num_walks=80 # Number of random walks to start at each node (default: 10)
# weighted=fairwalk
# weighted=unweighted
# pmodified=4
# weighted=random_walk_5_bndry_0.5_exp_2.0

# Running on a subset of the rice_subset dataset.
## Set which experiment to be ran to true to toggle it.
pmodified_experiment=false
c_d_experiment=false
rwl_bndry_exp_experiment=true #
synthetic_experiment=false
unweighted_experiment=false

test_links=$data/${dataset}/$dataset-test.links
train_links=$data/${dataset}/$dataset-train.links

test_link_ratio=0.5
train_link_ratio=0.5

if [ "$pmodified_experiment" = true ]; then
  echo $pmodified_experiment
  # Experimenting with different values for p
  for pmodified in 0.2 0.5 0.7 0.9 1.0; do
    # pmodified: Probability of using the modified graph (default: 1.0)
    python deepwalk --format edgelist \
                    --input $data/${dataset}/${dataset}.links \
                    --max-memory-data-size 0 \
                    --number-walks $num_walks \
                    --representation-size $d \
                    --walk-length 40 \
                    --window-size 10 \
                    --workers $num_workers \
                    --weighted $weighted  \
                    --output $data/${dataset}/${dataset}.pmodified_${pmodified}_embeddings_${weighted}_d$d \
                    --pmodified $pmodified \
                    --sensitive-attr-file $data/${dataset}/${dataset}.attr \
                    --test-links-file $test_links \
                    --train-links-file $train_links \
                    --test-links $test_link_ratio
  done
elif [ "$c_d_experiment" = true ]; then
  ## Experimenting with different values of c and d
  for c in 2 3 5 7 10 50 100 1000; do #2 3 5 7 10; do     # todo: figure out what c is
    for d in 32 64 92 128; do
      python deepwalk --format edgelist \
                      --input $data/${dataset}/${dataset}.links \
                      --max-memory-data-size 0 \
                      --number-walks $num_walks \
                      --representation-size $d \
                      --walk-length 40 \
                      --window-size 10 \
                      --workers $num_workers \
                      --output $data/${dataset}/${dataset}.embeddings_wconstant${c}_d$d \
                      --weighted constant_$c \
                      --sensitive-attr-file $data/${dataset}/${dataset}.attr \
                      --test-links-file $test_links \
                      --train-links-file $train_links \
                      --test-links $test_link_ratio
    done
  done
elif [ "$rwl_bndry_exp_experiment" = true ]; then
  for i in 1 2 3 4 5; do
    for dataset in 'rice_subset' 'twitter' 'synth2' 'synth3'; do
      for rwl in 5; do # 5 10 20
        for bndry in 0.5 0.7 0.9; do # 0.5 0.7 0.9 --> alpha
          for exponent in '1.0' '2.0' '3.0' '4.0'; do # '1.0' '2.0' '3.0' '4.0' --> p 
            for bndrytype in 'bndry'; do # 'bndry' 'revbndry'
              for d in 32; do # 32 64 92 128
                for method in 'random_walk' 'fairwalk' 'unweighted'; do #'random_walk' 'fairwalk' 'unweighted'
                  if [ $method = "unweighted" ]; then
                    full_method=${method}
                    test_links=$data/${dataset}/${method}/links/${method}_${dataset}_d$d-test-${test_link_ratio}_${i}.links
                    train_links=$data/${dataset}/${method}/links/${method}_${dataset}_d$d-train-${test_link_ratio}_${i}.links
                  else
                    full_method=${method}_${rwl}'_'$bndrytype'_'${bndry}'_exp_'${exponent}
                    test_links=$data/${dataset}/${method}/links/${full_method}-$dataset-test-${test_link_ratio}_${i}.links
                    train_links=$data/${dataset}/${method}/links/${full_method}-$dataset-train-${train_link_ratio}_${i}.links
                  fi
                  echo '   '
                  echo $data/${dataset}/${dataset}'  '$method 'rwl: ' $rwl 'bndry: ' $bndry 'exp: ' $exponent 'bndrytype: ' $bndrytype  'd: ' $d
                  python deepwalk --format edgelist \
                                  --input $data/${dataset}/${dataset}.links \
                                  --max-memory-data-size 0 \
                                  --number-walks $num_walks \
                                  --representation-size $d \
                                  --walk-length 40 \
                                  --window-size 10 \
                                  --workers $num_workers \
                                  --output $data/${dataset}/${method}/embeddings_${train_link_ratio}train${test_link_ratio}test/${dataset}.embeddings_${full_method}_d${d}_${i} \
                                  --weighted $full_method \
                                  --sensitive-attr-file $data/${dataset}/${dataset}.attr \
                                  --test-links-file $test_links \
                                  --train-links-file $train_links \
                                  --test-links $test_link_ratio
                done
              done
            done
          done
        done
      done
    done
  done
elif [ "$synthetic_experiment" = true ]; then
  nodes=500
  Pred=0.7
  Phom=0.025   # todo: phom/phet
  synthetic=synth2
  for i in ''; do #'2' '3' '4' '5'; do ???? what does i do?
    for Phet in 0.001; do #0.005 0.01 0.015
      filename=$data/$synthetic/synthetic_n${nodes}_Pred${Pred}_Phom${Phom}_Phet${Phet}
      method='unweighted'
      outfile=${filename}.embeddings_${method}_d${d}_$i
      echo ${i}'   '$method
      python deepwalk --format edgelist \
                      --input ${filename}.links \
                      --max-memory-data-size 0 \
                      --number-walks $num_walks \
                      --representation-size $d \
                      --walk-length 40 \
                      --window-size 10 \
                      --workers $num_workers \
                      --output $outfile \
                      --weighted $method \
                      --sensitive-attr-file ${filename}.attr \
                      --test-links-file $test_links \
                      --train-links-file $train_links \
                      --test-links $test_link_ratio
      for rwl in 5 10 20; do
        for bndry in 0.1 0.5 0.9; do
          for exponent in '1.0' '2.0' '3.0' '4.0'; do
            for bndrytype in 'bndry' 'revbndry'; do
              method='random_walk_'${rwl}'_'${bndrytype}'_'${bndry}'_exp_'${exponent}
              outfile=${filename}.embeddings_${method}_d${d}_$i
              echo ${i}'   '$method
              if test -f "$outfile"; then
                echo "exists."
              else
                python deepwalk --format edgelist \
                --input ${filename}.links \
                --max-memory-data-size 0 \
                --number-walks $num_walks \
                --representation-size $d \
                --walk-length 40 \
                --window-size 10 \
                --workers $num_workers \
                --output $outfile \
                --weighted $method \
                --sensitive-attr-file ${filename}.attr \
                --test-links-file $test_links \
                --train-links-file $train_links \
                --test-links $test_link_ratio
              fi
            done
          done
        done
      done
    done
  done
elif [ "$unweighted_experiment" = true ]; then
  python deepwalk --format edgelist \
                  --input $data/${dataset}/${dataset}.links \
                  --max-memory-data-size 0 \
                  --number-walks $num_walks \
                  --representation-size $d \
                  --walk-length 40 \
                  --window-size 10 \
                  --workers $num_workers \
                  --output $data/${dataset}/${dataset}.embeddings_unweighted \
                  --sensitive-attr-file $data/${dataset}/${dataset}.attr \
                  --test-links-file $test_links \
                  --train-links-file $train_links \
                  --test-links $test_link_ratio \
  done
fi