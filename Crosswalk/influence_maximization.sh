#! /bin/bash

# for synth2
python influence_maximization/fairinfMaximization.py \
        --method kmedoids \
        --dataset rice_subset
        
# for synth3
python influence_maximization/fairinfMaximization.py \
        --method kmedoids \
        --dataset twitter \
        --walking-algorithm