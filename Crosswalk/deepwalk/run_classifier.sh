
################################################################################################
# THIS FILE CREATES EMBEDDING FILES FOR THE FAIR, CROSS AND DEEPWALK EXPERIMENTS OF FIG. 6
# BASED ON THE RICE DATA SET.
# SEQUENTIALLY, IT RUNS THE CLASSIFICATION EXPERIMENT ON ALL THE FILES.
# THE RESULTS ARE SAVED IN THE OUTPUT FILES
################################################################################################

# SET TO TRUE IF YOU WANT TO DO AN ABLATION STUDY ON HYPERPARAMETER 'A'
ABLATION=false


# The dataset to be used in deepwalk to generate node representations.
dataset='rice_subset'

# Running parameters
data=../data # Directory of the dataset.
d=32 # Dimensionality of the latent space.
num_workers=1 # Number of parallel processes. (default: 1)
num_walks=80 # Number of random walks to start at each node (default: 10)
indices=5 # number of files created per method

# creates 5 files for each hyperparameter 'a'
if [ "$ABLATION" = true ]; then
    for a in 0.0 0.2 0.4 0.6 0.8 1.0; 
    do    
        method=random_walk_5_bndry_${a}_exp_2.0
        for i in $(seq 1 $indices)
        do
        python3 deepwalk --format edgelist \
                            --input $data/${dataset}/${dataset}.links \
                            --max-memory-data-size 0 \
                            --number-walks $num_walks \
                            --representation-size $d \
                            --walk-length 40 \
                            --window-size 10 \
                            --workers $num_workers \
                            --weighted $method  \
                            --output $data/${dataset}/ablation/rice_subset.embeddings_${method}_${i} \
                            --sensitive-attr-file $data/${dataset}/${dataset}.attr
        done
    done
    # runs the python classification file with ablation
    python3 ../classifier/main_fixed.py --ablation True --nfiles $indices
else
    for method in 'random_walk_5_bndry_0.5_exp_2.0' 'unweighted' 'fairwalk'; do
        for i in $(seq 1 $indices); do
        python3 deepwalk --format edgelist \
                            --input $data/${dataset}/${dataset}.links \
                            --max-memory-data-size 0 \
                            --number-walks $num_walks \
                            --representation-size $d \
                            --walk-length 40 \
                            --window-size 10 \
                            --workers $num_workers \
                            --weighted $method  \
                            --output $data/${dataset}/rice_subset.embeddings_${method}_${i} \
                            --sensitive-attr-file $data/${dataset}/${dataset}.attr
            done
        done
    # runs the python classification file on the created files
    python3 ../classifier/main_fixed.py --nfiles $indices
fi
