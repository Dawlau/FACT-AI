#! /bin/bash -l

datasets=("rice_subset" "synth2" "synth3" "twitter")
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) # Directory of script
data=${SCRIPT_DIR}/../datahub # Directory of the dataset.
budget=40
exponent_values=(1.0 2.0 4.0 6.0 8.0)
alpha_values=(0.1 0.3 0.5 0.7 0.9)

declare -A dataset_to_alpha=(["rice_subset"]=0.5 ["synth2"]=0.7 ["synth3"]=0.7 ["twitter"]=0.5)
declare -A dataset_to_p=(["rice_subset"]=4.0 ["synth2"]=4.0 ["synth3"]=4.0 ["twitter"]=2.0)

for dataset in ${datasets[@]}; do

	 python influence_maximization/fairinfMaximization.py \
	 		--method greedy \
	 		--dataset $dataset \
	 		--budget $budget

	 echo "Done running the greedy algorithm for" $dataset

	for walking_algorithm in "unweighted" "fairwalk"; do

		python influence_maximization/fairinfMaximization.py \
			--method kmedoids \
			--walking_algorithm $walking_algorithm \
			--dataset $dataset \
			--budget $budget

		echo "Done running the kmedoids algorithm for" $dataset "and walking algorithm" $walking_algorithm
	done


	python influence_maximization/fairinfMaximization.py \
		--method kmedoids \
		--walking_algorithm random_walk \
		--dataset $dataset \
		--alpha ${dataset_to_alpha[$dataset]} \
		--exponent_p ${dataset_to_p[$dataset]} \
		--budget $budget

	echo "Done running the kmedoids algorithm for" $dataset "and walking algorithm Crosswalk for parameters alpha" ${dataset_to_alpha[$dataset]} "and exponent" ${dataset_to_p[$dataset]}

	if [ $dataset = "rice_subset" ] || [ $dataset = "synth2" ] || [ $dataset = "soft_rice_subset" ]; then

		attr_file=$(ls ${data}/${dataset} | grep "\.attr")
		links_file=$(ls ${data}/${dataset} | grep "\.links")

		attr_file=${data}/${dataset}/${attr_file}
		links_file=${data}/${dataset}/${links_file}

		python aae.py \
			--attr_filename $attr_file \
			--links_filename $links_file \
			--dataset $dataset
	fi
done

datasets=("rice_subset" "twitter")
exponent_values=(1.0 2.0 4.0 6.0 8.0)
alpha_values=(0.1 0.3 0.5 0.7 0.9)

for dataset in ${datasets[@]}; do
	for alpha in ${alpha_values[@]}; do
		for p in ${exponent_values[@]}; do
			python influence_maximization/fairinfMaximization.py \
					--method kmedoids \
					--walking_algorithm random_walk \
					--dataset $dataset \
					--alpha ${alpha} \
					--exponent_p ${p} \
					--budget $budget
		done
	done
done
