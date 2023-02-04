#! /bin/bash

budget=40
datasets=("soft_rice_subset" "soft_synth2" "soft_synth3")
exponent_values=(1.0 4.0 8.0)
alpha_values=(0.5 0.7)

for dataset in ${datasets[@]}; do
	for alpha in ${alpha_values[@]}; do
		for p in ${exponent_values[@]}; do
      python fairinfMaximization.py \
          --method kmedoids \
          --walking_algorithm soft_random_walk \
          --dataset $dataset \
          --alpha ${alpha} \
          --exponent_p ${p} \
          --budget $budget
    done
	done
done
