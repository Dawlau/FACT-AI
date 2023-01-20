# The dataset to be used in deepwalk to generate node representations.
dataset='twitter'

# Running parameters
data=../data # Directory of the dataset.
d=32 # Dimensionality of the latent space.
num_workers=1 # Number of parallel processes. (default: 1)
num_walks=80 # Number of random walks to start at each node (default: 10)
# weighted=fairwalk
# weighted=unweighted
# pmodified=4
weighted=random_walk_5_bndry_0.7_exp_4.0

# Running on a subset of the rice_subset dataset.
## Set which experiment to be ran to true to toggle it.
pmodified_experiment=false
c_d_experiment=false
rwl_bndry_exp_experiment=false
synthetic_experiment=false
unweighted_experiment=false

python deepwalk --format edgelist \
                --input $data/${dataset}/sample_4000.links \
                --max-memory-data-size 0 \
                --number-walks $num_walks \
                --representation-size $d \
                --walk-length 40 \
                --window-size 10 \
                --workers $num_workers \
                --weighted $weighted  \
                --output $data/${dataset}/${dataset}.pmodified_${pmodified}_embeddings_${weighted}_d$d \
                --sensitive-attr-file $data/${dataset}/sample_4000.attr \
                # --sensitive-attr-file $data/${dataset}/synthetic_3g_n500_Pred0.6_Pblue0.25_Prr0.025_Pbb0.025_Pgg0.025_Prb0.001_Prg0.0005_Pbg0.0005.attr \
                # --sensitive-attr-file $data/${dataset}/synthetic_n500_Pred0.7_Phom0.025_Phet0.001.attr \
                # --sensitive-attr-file $data/${dataset}/${dataset}.attr \


if [ "$pmodified_experiment" = true ]; then
  echo $pmodified_experiment
  ## Experimenting with different values for p
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
                    --sensitive-attr-file $data/${dataset}/${dataset}.attr
  done
elif [ "$c_d_experiment" = true ]; then
  ## Experimenting with different values of c and d
  for c in 50 100 1000; do #2 3 5 7 10; do     # todo: figure out what c is
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
                      --sensitive-attr-file $data/${dataset}/${dataset}.attr
    done
  done
elif [ "$rwl_bndry_exp_experiment" = true ]; then
  for rwl in 5; do # 5 10 20
    for bndry in 0.5; do # 0.5 0.7 0.9
      for exponent in '2.0' '4.0'; do # '1.0' '2.0' '3.0' '4.0'
        for bndrytype in 'bndry'; do # 'bndry' 'revbndry'
          method='random_walk_'${rwl}'_'$bndrytype'_'${bndry}'_exp_'${exponent}
          echo '   '
          echo $data/${dataset}/${dataset}'  '$method
          python deepwalk --format edgelist \
                          --input $data/${dataset}/${dataset}.links \
                          --max-memory-data-size 0 \
                          --number-walks $num_walks \
                          --representation-size $d \
                          --walk-length 40 \
                          --window-size 10 \
                          --workers $num_workers \
                          --output $data/${dataset}/${dataset}.embeddings_${method}_d$d \
                          --weighted $method \
                          --sensitive-attr-file $data/${dataset}/${dataset}.attr
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
    for Phet in 0.001 0.005 0.01 0.015; do
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
                      --sensitive-attr-file ${filename}.attr
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
                --sensitive-attr-file ${filename}.attr
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
                  --sensitive-attr-file $data/${dataset}/${dataset}.attr
fi

# echo "Running last experiment"

# synthetic 3 layers
# nodes=500
# Pred=0.7
# Phet=0.001
# Phom=0.025
# synthetic=synthetic_3layers
# for i in ''; do #'2' '3' '4' '5'; do ???? what does i do?
#   filename=$data/$synthetic/${synthetic}_n${nodes}_Pred${Pred}_Phom${Phom}_Phet${Phet}
#   method='unweighted'
#   echo "filename: ${filename}, method: ${method}"
#   python deepwalk --format edgelist \
#                   --input ${filename}.links \
#                   --max-memory-data-size 0 \
#                   --number-walks $num_walks \
#                   --representation-size $d \
#                   --walk-length 40 \
#                   --window-size 10 \
#                   --workers $num_workers \
#                   --output ${filename}.embeddings_${method}_d${d} \
#                   --weighted $method \
#                   --sensitive-attr-file ${filename}.attr
#   for rwl in 5; do # 5 10 20
#     for bndry in 0.5; do # 0.5 0.7 0.9
#       for exponent in '2.0' '4.0'; do # '1.0' '2.0' '3.0' '4.0'
#         for bndrytype in 'bndry'; do # 'bndry' 'revbndry'
#           method='random_walk_'${rwl}'_'$bndrytype'_'${bndry}'_exp_'${exponent}
#           echo ""
#           echo "filename: ${filename}, method: ${method}"
#           python deepwalk --format edgelist \
#                           --input ${filename}.links \
#                           --max-memory-data-size 0 \
#                           --number-walks $num_walks \
#                           --representation-size $d \
#                           --walk-length 40 \
#                           --window-size 10 \
#                           --workers $num_workers \
#                           --output ${filename}.embeddings_${method}_d${d} \
#                           --weighted $method \
#                          --sensitive-attr-file ${filename}.attr
#         done
#       done
#     done
#   done
# done


#prb=0.7
#pbr=0.7

#nodes=500
#Pred=0.7
#Phom=0.025
#
#
#for i in ''; do #'2' '3' '4' '5'; do
#	for Phet in 0.001 0.005 0.01 0.015; do # 0.001 0.005 0.01 0.015 0.02 0.025; do
#		filename=synthetic/synthetic_n${nodes}_Pred${Pred}_Phom${Phom}_Phet${Phet}
#		#method='unweighted'
#		#outfile=${filename}.embeddings_${method}_d${d}_$i
#		#echo ${i}'   '$method
#		#python deepwalk --format edgelist --input ${filename}.links --max-memory-data-size 0 --number-walks 80 --representation-size $d --walk-length 40 --window-size 10 --workers 30 --output $outfile --weighted $method --sensitive-attr-file ${filename}.attr
#		for rwl in 5; do #5 10 20; do
#			for bndry in 0.1 0.5 0.9; do #0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do #0.2 0.5 0.7 0.9; do
#				for exponent in '5.0'; do #'2.0' '5.0' '8.0'; do #1.0 1.25 1.5 1.75 2.0 4.0 6.0 8.0; do #0.5 1.0 2.0; do
#					for bndrytype in 'bndry'; do # 'revbndry'; do
#						method='random_walk_'${rwl}'_'${bndrytype}'_'${bndry}'_exp_'${exponent}
#						outfile=${filename}.embeddings_${method}_d${d}_$i
#						#echo '   '
#						#echo $filename'  '$method
#						echo ${i}'   '$method
#						if test -f "$outfile"; then
#							echo "exists."
#						else
#							python deepwalk --format edgelist --input ${filename}.links --max-memory-data-size 0 --number-walks 80 --representation-size $d --walk-length 40 --window-size 10 --workers 30 --output $outfile --weighted $method --sensitive-attr-file ${filename}.attr
#							#						#python deepwalk --format edgelist --input ${filename}.links --max-memory-data-size 0 --number-walks 160 --representation-size $d --walk-length 40 --window-size 10 --workers 30 --output ${filename}.pmodified_${pmodified}_embeddings_${method}_d$d --weighted $method --pmodified $pmodified --sensitive-attr-file ${filename}.attr
#						fi
#					done
#				done
#			done
#		done
#	done
#done


#filename='data/rice_subset/rice_subset'
#for rwl in 5; do # 5 10 20; do
#	for bndry in 0.5 0.7 0.9; do #0.2 0.5 0.7 0.9; do
#		for exponent in '1.0' '2.0' '3.0' '4.0'; do #0.5 1.0 2.0; do
#			for bndrytype in 'bndry'; do # 'bndry' 'revbndry'; do
#				method='random_walk_'${rwl}'_'$bndrytype'_'${bndry}'_exp_'${exponent}
#				echo '   '
#				echo $filename'  '$method
#				python deepwalk --format edgelist --input ${filename}.links --max-memory-data-size 0 --number-walks 80 --representation-size $d --walk-length 40 --window-size 10 --workers 30 --output ${filename}.embeddings_${method}_d$d --weighted $method --sensitive-attr-file ${filename}.attr
#			done
#		done
#	done
#done




# different num wlks: 160 vs 80
#python deepwalk --format edgelist --input data/${dataset}/${dataset}.links --max-memory-data-size 0 --number-walks 160 --representation-size $d --walk-length 40 --window-size 10 --workers 30 --output data/test/rice_subset.embeddings_pch_0.9_d${d}__2 --weighted pch_0.9 --sensitive-attr-file data/${dataset}/${dataset}.attr
#python deepwalk --format edgelist --input data/${dataset}/${dataset}.links --max-memory-data-size 0 --number-walks 80 --representation-size $d --walk-length 40 --window-size 10 --workers 30 --output data/test/rice_subset.embeddings_pch_0.9_d${d}__1 --weighted pch_0.9 --sensitive-attr-file data/${dataset}/${dataset}.attr

#for d in 64 92 128; do
#	python deepwalk --format edgelist --input data/${dataset}/${dataset}.links --max-memory-data-size 0 --number-walks 80 --representation-size $d --walk-length 40 --window-size 10 --workers 30 --output data/${dataset}/${dataset}.embeddings_prb_${prb}_pbr_${pbr}_d$d --weighted prb_${prb}_pbr_${pbr} --sensitive-attr-file data/${dataset}/${dataset}.attr

##for dataset in 'rice_subset' 'sample_1000' 'sample_4000'; do
#for dataset in 'rice_subset'; do # 'sample_1000' 'sample_4000'; do
#	python deepwalk --format edgelist --input data/${dataset}/${dataset}.links --max-memory-data-size 0 --number-walks 80 --representation-size $d --walk-length 40 --window-size 10 --workers 30 --output data/${dataset}/${dataset}.embeddings_random_d$d --weighted random --sensitive-attr-file data/${dataset}/${dataset}.attr
#for psc in 0.9 0.7 0.5 0.2; do
#	echo $dataset
#	python deepwalk --format edgelist --input data/${dataset}/${dataset}.links --max-memory-data-size 0 --number-walks 80 --representation-size $d --walk-length 40 --window-size 10 --workers 30 --output data/${dataset}/${dataset}.embeddings_smartshortcut_${psc}_d$d --weighted smartshortcut_${psc} --sensitive-attr-file data/${dataset}/${dataset}.attr
#done
#	python deepwalk --format edgelist --input data/${dataset}/${dataset}.links --max-memory-data-size 0 --number-walks 80 --representation-size $d --walk-length 40 --window-size 10 --workers 30 --output data/${dataset}/${dataset}.embeddings_prb_${prb}_pbr_${pbr}_d$d --weighted prb_${prb}_pbr_${pbr} --sensitive-attr-file data/${dataset}/${dataset}.attr

# python deepwalk --format edgelist --input data/${dataset}/${dataset}.links --max-memory-data-size 0 --number-walks 80 --representation-size $d --walk-length 40 --window-size 10 --workers 30 --output data/${dataset}/${dataset}.embeddings_wexpandconstant${c}_d$d --weighted expandar_constant_$c --sensitive-attr-file data/${dataset}/${dataset}.attr

#	python deepwalk --format edgelist --input data/${dataset}/${dataset}.links --max-memory-data-size 0 --number-walks 80 --representation-size $d --walk-length 40 --window-size 10 --workers 30 --output data/${dataset}/${dataset}.embeddings_wrb_${wrb}_wbr_1_d$d --weighted rb_${wrb}_br_1 --sensitive-attr-file data/${dataset}/${dataset}.attr
#	python deepwalk --format edgelist --input data/${dataset}/${dataset}.links --max-memory-data-size 0 --number-walks 80 --representation-size $d --walk-length 40 --window-size 10 --workers 30 --output data/${dataset}/${dataset}.embeddings_unweighted --sensitive-attr-file data/${dataset}/${dataset}.attr
#done


