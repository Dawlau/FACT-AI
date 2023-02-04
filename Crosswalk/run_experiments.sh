#! /bin/bash -l

# RGenerate Node Embeddings
bash deepwalk/run.sh
bash deepwalk/run_soft_deepwalk.sh

# Classifier
bash classifier/run.sh
bash classifier/run_soft_classifier.sh
python classifier/visualize_results.py

# Influence Maximization
bash influence_maximization/run.sh
bash influence_maximization/run_soft_infmax.sh
python influence_maximization/visualize_results.py

# Link Prediction
python link_prediction/main.py
python link_prediction/visualize_link_prediction.py

# Statistics
python statistics/regularization.py

# TSNE
python tsne/main.py

# Visualisation
cd visualisation && python visualisation.py && cd ..
